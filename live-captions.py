#!/usr/bin/env python3
import argparse, queue, sys, time, threading, os, platform
import sounddevice as sd
from faster_whisper import WhisperModel
import webrtcvad, numpy as np
import signal
import scipy.signal as sig
from collections import deque
from time import perf_counter
from log import log_info, log_warn, log_error, log_debug

# ── User-facing defaults ─────────────────────────────────────────────────────────────────────────
MIC_DEVICE_NAME = "MacBook Pro Microphone"
SYS_DEVICE_NAME = "BlackHole 2ch"

DEFAULT_MODEL = "base.en"  # use --model medium.en on Apple Silicon for higher accuracy
DEFAULT_END_SILENCE_MS = 260
DEFAULT_MAX_SEGMENT_MS = 1000
DEFAULT_NOISE_GATE_DB = -52.0
DEFAULT_SPEECH_THRESHOLD = 1  # WebRTC VAD aggressiveness (0..3)
DEFAULT_BEAM_SIZE = 2

# ── Streaming/VAD constants (balanced) ───────────────────────────────────────────────────────────
FRAME_MS = 20
LOOKBACK_MS = 280
TAIL_MS = 220
ENTER_VOICED_MIN_FRAMES = 2

# Segment acceptance guards
MIN_VOICED_MS = 80
VOICED_RATIO_MIN = 0.22
SNR_MARGIN_DB = 3.5
ADAPTIVE_GATE_MARGIN_DB = 4.5

# Noise tracker
NOISE_FLOOR_DB_INIT = -60.0
NOISE_EMA_ALPHA = 0.96

TARGET_SR = 16000
BYTES_PER_SAMPLE = 2

# Rolling context for better continuations
CTX_CHARS = 120
ctx_lock = threading.Lock()
rolling_ctx = ""  # last ~120 chars of accepted text


# ── DSP helpers ──────────────────────────────────────────────────────────────────────────────────
def rms_dbfs(yf: np.ndarray) -> float:
    eps = 1e-12
    rms = np.sqrt(np.mean(yf * yf) + eps)
    return 20.0 * np.log10(rms + eps)


class StreamingHPF:
    """2nd-order Butterworth high-pass (~100 Hz), stateful and fast."""

    def __init__(self, fc=100.0, sr=16000):
        sos = sig.butter(2, fc / (sr / 2), btype="highpass", output="sos")
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2), dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sig.sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32, copy=False)


def pre_emphasis_vec(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if y.size == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    out[1:] = y[1:] - coeff * y[:-1]
    return out


def apply_agc_once(y: np.ndarray, target_db=-20.0, max_gain_db=12.0) -> np.ndarray:
    lvl = rms_dbfs(y)
    gain_db = np.clip(target_db - lvl, -0.1, max_gain_db)
    return np.clip(y * (10 ** (gain_db / 20.0)), -1.0, 1.0).astype(np.float32, copy=False)


def vad_vote_10_20_30(y: np.ndarray, vad: webrtcvad.Vad, sr=16000) -> bool:
    n10 = sr // 100  # 160 samples
    n = len(y)

    def check(buf):
        b = (np.clip(buf, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()
        return vad.is_speech(b, sr)

    if n == 3 * n10:
        votes = sum(check(y[i * n10:(i + 1) * n10]) for i in range(3))
        return votes >= 2
    if n == 2 * n10:
        return check(y[:n10]) or check(y[n10:2 * n10])
    if n == n10:
        return check(y)
    m = (n // n10) * n10 or n10
    yy = y[:m] if n >= m else np.pad(y, (0, m - n))
    return check(yy)


# ── Audio helpers ────────────────────────────────────────────────────────────────────────────────
def list_input_devices():
    devices = sd.query_devices()
    return [{"index": i, "name": d.get("name", f"Device {i}")}
            for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]


def resample_16k(frame_bytes_dev: bytes, device_sr: int, out_len: int) -> np.ndarray:
    x = np.frombuffer(frame_bytes_dev, dtype=np.int16).astype(np.float32) / 32768.0
    if device_sr == TARGET_SR:
        y = x
    else:
        y = sig.resample_poly(x, TARGET_SR, device_sr).astype(np.float32)
    if len(y) != out_len:
        if len(y) > out_len:
            y = y[:out_len]
        else:
            y = np.pad(y, (0, out_len - len(y)))
    return y


# ── ASR worker (decoupled from capture) ──────────────────────────────────────────────────────────
def asr_worker(seg_q: "queue.Queue", stop_evt: threading.Event,
               model: WhisperModel, timestamps: bool, base_beam: int):
    global rolling_ctx
    while not stop_evt.is_set() or not seg_q.empty():
        try:
            item = seg_q.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            t_start = perf_counter()
            seg, meta = item if isinstance(item, tuple) else (item, {})
            wait_ms = (t_start - meta.get("t_enqueue", t_start)) * 1000.0
            len_ms = meta.get("len_ms", int(len(seg) / TARGET_SR * 1000))

            # Pre-emphasis + AGC on the finalized segment (cheap, helps fricatives)
            seg = pre_emphasis_vec(seg)
            seg = apply_agc_once(seg)

            # Dynamic beam: speed up when backlog builds
            qsz = seg_q.qsize()
            beam = 1 if qsz >= 3 else base_beam

            # Read short context, then release lock before decode
            with ctx_lock:
                init_prompt = rolling_ctx[-CTX_CHARS:]

            t_decode0 = perf_counter()
            it, _info = model.transcribe(
                seg,
                beam_size=beam,
                temperature=0.0,
                vad_filter=False,
                word_timestamps=False,
                language="en",
                task="transcribe",
                condition_on_previous_text=False,
                initial_prompt=init_prompt,
                compression_ratio_threshold=2.6,
                log_prob_threshold=-0.65,
                no_speech_threshold=0.85,
            )
            segments = list(it)  # force materialization to measure true time
            t_end = perf_counter()
            decode_ms = (t_end - t_decode0) * 1000.0
            behind_ms = wait_ms + decode_ms
            log_debug(
                f"[asr] len={len_ms}ms wait={wait_ms:.0f}ms decode={decode_ms:.0f}ms behind≈{behind_ms:.0f}ms beam={beam}")

            # Filter + join once; update rolling context under lock
            out_chunks = []
            for s in segments:
                if (getattr(s, "no_speech_prob", 0) > 0.85) or \
                        (getattr(s, "avg_logprob", 0) < -0.65) or \
                        (getattr(s, "compression_ratio", 0) > 2.6):
                    continue
                out_chunks.append(s.text)
            out_text = " ".join(out_chunks).strip()

            if out_text:
                with ctx_lock:
                    rolling_ctx = (rolling_ctx + " " + out_text).strip()
                    if len(rolling_ctx) > CTX_CHARS:
                        rolling_ctx = rolling_ctx[-CTX_CHARS:]

                if timestamps:
                    now_ts = time.strftime("%H:%M:%S")
                    print(f"[{now_ts}] {out_text}", flush=True)
                else:
                    print(out_text, flush=True)

        except Exception as e:
            log_error(f"[stt] transcribe error: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Live captions from mic or system audio.")
    p.add_argument("--source", type=str, help="mic | sys | device index")
    p.add_argument("--profile", choices=["fast", "default", "slow"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--end-silence-ms", type=int, default=DEFAULT_END_SILENCE_MS)
    p.add_argument("--max-segment-ms", type=int, default=DEFAULT_MAX_SEGMENT_MS)
    p.add_argument("--noise-gate-db", type=float, default=DEFAULT_NOISE_GATE_DB)
    p.add_argument("--speech-threshold", type=int, default=DEFAULT_SPEECH_THRESHOLD,
                   choices=[0, 1, 2, 3])
    p.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    p.add_argument("--timestamps", action="store_true")
    args = p.parse_args()

    # Profiles
    if args.profile == "fast":
        prof = dict(model="base.en", end_silence_ms=180, max_segment_ms=500,
                    noise_gate_db=-42.0, speech_threshold=2, beam_size=1)
    elif args.profile == "slow":
        prof = dict(model="large-v2", end_silence_ms=300, max_segment_ms=5000,
                    noise_gate_db=-55.0, speech_threshold=0, beam_size=8)
    else:
        prof = dict(model=DEFAULT_MODEL, end_silence_ms=DEFAULT_END_SILENCE_MS,
                    max_segment_ms=DEFAULT_MAX_SEGMENT_MS, noise_gate_db=DEFAULT_NOISE_GATE_DB,
                    speech_threshold=DEFAULT_SPEECH_THRESHOLD, beam_size=DEFAULT_BEAM_SIZE)

    # Apply overrides
    model_name = args.model if args.model != DEFAULT_MODEL else prof["model"]
    end_silence_ms = args.end_silence_ms if args.end_silence_ms != DEFAULT_END_SILENCE_MS else prof[
        "end_silence_ms"]
    max_segment_ms = args.max_segment_ms if args.max_segment_ms != DEFAULT_MAX_SEGMENT_MS else prof[
        "max_segment_ms"]
    noise_gate_db = args.noise_gate_db if args.noise_gate_db != DEFAULT_NOISE_GATE_DB else prof[
        "noise_gate_db"]
    speech_threshold = args.speech_threshold if args.speech_threshold != DEFAULT_SPEECH_THRESHOLD else \
        prof["speech_threshold"]
    beam_size = args.beam_size if args.beam_size != DEFAULT_BEAM_SIZE else prof["beam_size"]

    # Device discovery
    devices = list_input_devices()
    mbp_idx = next((d["index"] for d in devices if MIC_DEVICE_NAME in d["name"]), None)
    bh_idx = next((d["index"] for d in devices if SYS_DEVICE_NAME in d["name"]), None)

    if args.source is None:
        device_index = None
    elif args.source == "mic":
        device_index = mbp_idx
    elif args.source == "sys":
        device_index = bh_idx
    else:
        try:
            device_index = int(args.source)
        except ValueError:
            device_index = None

    if device_index is None:
        log_error("Please specify a valid --source:\n")
        log_info(f"mic : {MIC_DEVICE_NAME} ({'not found' if mbp_idx is None else mbp_idx})")
        log_info(f"sys : {SYS_DEVICE_NAME} ({'not found' if bh_idx is None else bh_idx})")
        for d in devices:
            log_info(f"{d['index']:>3} : {d['name']}")
        sys.exit(1)

    # Print params
    log_info("Live captions")
    log_info(f"--source           : {args.source}")
    log_info(f"--profile          : {args.profile or 'default'}")
    log_info(f"--model            : {model_name}")
    log_info(f"--end-silence-ms   : {end_silence_ms} ms")
    log_info(f"--max-segment-ms   : {max_segment_ms} ms")
    log_info(f"--noise-gate-db    : {noise_gate_db} dBFS")
    log_info(f"--speech-threshold : {speech_threshold}")
    log_info(f"--beam-size        : {beam_size}")
    log_info(f"--timestamps       : {'on' if args.timestamps else 'off'}\n")

    # Init audio
    device_sr = int(sd.query_devices(device_index)["default_samplerate"])
    samples_per_frame_dev = int(device_sr * FRAME_MS / 1000)
    samples_per_frame_16k = int(TARGET_SR * FRAME_MS / 1000)
    frame_bytes_dev = samples_per_frame_dev * BYTES_PER_SAMPLE

    q_audio = queue.Queue(maxsize=512)
    max_buf_ms = 1000
    max_buf_bytes = int(device_sr * max_buf_ms / 1000) * BYTES_PER_SAMPLE

    vad = webrtcvad.Vad(speech_threshold)

    # --- Model init (robust: try HF IDs, fall back to base/base.en) ---
    def init_model_resilient(preferred_ids, cpu_threads):
        tried = []
        for model_id in preferred_ids:
            for dev, ctype, kw in [
                ("cuda", "float16", {}),  # if you actually have NVIDIA GPU
                ("cpu", "int8", {"cpu_threads": cpu_threads}),  # fast CPU path
                ("cpu", "float32", {"cpu_threads": cpu_threads}),  # last resort
            ]:
                try:
                    m = WhisperModel(model_id, device=dev, compute_type=ctype, **kw)
                    log_info(f"Backend: {dev} ({ctype}), model: {model_id}")
                    return m
                except Exception as e:
                    tried.append(f"{model_id}@{dev}/{ctype}: {e}")
        raise RuntimeError("Failed to init any model. Tried → " + " | ".join(tried))

    cpu_threads = max(1, (os.cpu_count() or 4) - 1)

    # Build a candidate list: CLI choice first, then common fallbacks
    candidate_models = [
        model_name,  # whatever came from CLI/profile
        "Systran/faster-whisper-small.en",  # faster-whisper canonical repo
        "small.en",  # alias
        "Systran/faster-whisper-base.en",
        "base.en",
        "base",
    ]

    model = init_model_resilient(candidate_models, cpu_threads)

    # Streaming DSP
    hpf = StreamingHPF(fc=100.0, sr=TARGET_SR)

    # Capture state
    buf = bytearray()
    in_speech = False
    silence_frames = 0
    segment_len_frames = 0
    segment_voiced_frames = 0
    enter_voiced_streak = 0

    lookback_frames = LOOKBACK_MS // FRAME_MS
    lookback_16k = deque(maxlen=lookback_frames)
    tail_frames = TAIL_MS // FRAME_MS
    trailing_tail_needed = 0

    current_segment = []
    dropped_frames = 0
    last_warn = 0.0
    noise_floor_db = NOISE_FLOOR_DB_INIT

    # ASR worker
    seg_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
    stop_evt = threading.Event()
    worker = threading.Thread(
        target=asr_worker,
        args=(seg_q, stop_evt, model, args.timestamps, beam_size),
        daemon=True,
    )
    worker.start()

    def callback(indata, _frames, _time, status):
        nonlocal dropped_frames, last_warn
        try:
            if status:
                now = time.time()
                if now - last_warn > 5:
                    log_warn(f"[audio] status: {status}")
                    last_warn = now
            if q_audio.full():
                try:
                    _ = q_audio.get_nowait()
                    dropped_frames += 1
                except queue.Empty:
                    pass
            q_audio.put_nowait(bytes(indata))
        except Exception as e:
            now = time.time()
            if now - last_warn > 5:
                log_warn(f"[audio] callback error: {e}")
                last_warn = now

    device_info = sd.query_devices(device_index)
    log_warn(f"Using device: {device_info['name']} ({device_index})")

    try:
        stream = sd.RawInputStream(
            dtype="int16", channels=1, samplerate=device_sr,
            blocksize=samples_per_frame_dev,  # 20ms
            callback=callback, device=device_index)
    except Exception as e:
        log_error(f"Failed to open audio stream: {e}")
        stop_evt.set();
        worker.join(timeout=1.0)
        sys.exit(2)

    with stream:
        log_warn("Listening... Press Ctrl+C to stop.")
        last_stats = time.time()
        last_telemetry = time.time()
        try:
            while True:
                try:
                    chunk = q_audio.get(timeout=1.0)
                except queue.Empty:
                    continue

                buf.extend(chunk)
                if len(buf) > max_buf_bytes:
                    buf[:] = buf[-max_buf_bytes:]

                if len(buf) < frame_bytes_dev:
                    continue

                frame_dev = bytes(buf[:frame_bytes_dev])
                del buf[:frame_bytes_dev]

                # Resample + cheap DSP for VAD
                f16 = resample_16k(frame_dev, device_sr, samples_per_frame_16k)
                y = hpf.process(f16)

                # VAD + adaptive gate
                vad_voiced = vad_vote_10_20_30(y, vad, sr=TARGET_SR)
                level_db = rms_dbfs(y)
                if not vad_voiced:
                    noise_floor_db = (NOISE_EMA_ALPHA * noise_floor_db) + (
                            (1 - NOISE_EMA_ALPHA) * level_db)
                gate_db = max(noise_gate_db, noise_floor_db + ADAPTIVE_GATE_MARGIN_DB)
                voiced = vad_voiced and (level_db >= gate_db)

                if not in_speech:
                    lookback_16k.append(y)
                    if voiced:
                        enter_voiced_streak += 1
                        if enter_voiced_streak >= ENTER_VOICED_MIN_FRAMES:
                            in_speech = True
                            silence_frames = 0
                            segment_len_frames = 0
                            segment_voiced_frames = 0
                            current_segment = list(lookback_16k)
                            trailing_tail_needed = 0
                    else:
                        enter_voiced_streak = 0
                else:
                    segment_len_frames += 1
                    if voiced:
                        current_segment.append(y)
                        segment_voiced_frames += 1
                        silence_frames = 0
                        trailing_tail_needed = tail_frames
                    else:
                        silence_frames += 1
                        if trailing_tail_needed > 0:
                            current_segment.append(y)
                            trailing_tail_needed -= 1

                    end_on_silence = (silence_frames * FRAME_MS) >= end_silence_ms
                    force_on_timeout = (segment_len_frames * FRAME_MS) >= max_segment_ms
                    if end_on_silence or force_on_timeout:
                        if current_segment:
                            seg = np.concatenate(current_segment, axis=0).astype(np.float32)
                            voiced_ms = segment_voiced_frames * FRAME_MS
                            voiced_ratio = segment_voiced_frames / max(1, segment_len_frames)
                            seg_db = rms_dbfs(seg)

                            if (voiced_ms >= MIN_VOICED_MS) and (
                                    voiced_ratio >= VOICED_RATIO_MIN) and (
                                    seg_db >= (noise_floor_db + SNR_MARGIN_DB)):
                                try:
                                    qs_now = seg_q.qsize()
                                    # Backpressure: if backlog builds, drop weaker segments
                                    if qs_now >= 6 and (
                                            voiced_ratio < 0.40 or seg_db < noise_floor_db + 4.0):
                                        log_debug(
                                            f"[drop:backpressure] len={segment_len_frames * FRAME_MS}ms vr={voiced_ratio:.2f} seg={seg_db:.1f}dB nf={noise_floor_db:.1f}dB qs={qs_now}")
                                    else:
                                        meta = {
                                            "len_ms": segment_len_frames * FRAME_MS,
                                            "voiced_ms": segment_voiced_frames * FRAME_MS,
                                            "voiced_ratio": voiced_ratio,
                                            "seg_db": seg_db,
                                            "noise_floor_db": noise_floor_db,
                                            "gate_db": gate_db,
                                            "t_enqueue": perf_counter(),
                                        }
                                        seg_q.put_nowait((seg, meta))
                                        log_debug(
                                            f"[enqueue] len={meta['len_ms']}ms vr={voiced_ratio:.2f} seg={seg_db:.1f}dB nf={noise_floor_db:.1f}dB gate={gate_db:.1f}dB qs={seg_q.qsize()}")
                                except queue.Full:
                                    log_warn("[asr] segment queue full; dropping one segment")
                            else:
                                reason = []
                                if voiced_ms < MIN_VOICED_MS: reason.append("min_voiced")
                                if voiced_ratio < VOICED_RATIO_MIN: reason.append("ratio")
                                if seg_db < (noise_floor_db + SNR_MARGIN_DB): reason.append("snr")
                                log_debug(
                                    f"[skip:{'+'.join(reason) or 'unknown'}] vm={voiced_ms}ms vr={voiced_ratio:.2f} seg={seg_db:.1f}dB nf={noise_floor_db:.1f}dB gate={gate_db:.1f}dB")

                        # reset
                        in_speech = False
                        silence_frames = 0
                        segment_len_frames = 0
                        enter_voiced_streak = 0
                        current_segment = []
                        lookback_16k.clear()
                        trailing_tail_needed = 0

                # Periodic diagnostics
                now = time.time()
                if now - last_stats > 10:
                    if dropped_frames:
                        log_debug(f"dropped frames in last 10s: {dropped_frames}")
                        dropped_frames = 0
                    last_stats = now
                if now - last_telemetry > 1.0:
                    try:
                        qa = q_audio.qsize()
                    except Exception:
                        qa = -1
                    try:
                        qs = seg_q.qsize()
                    except Exception:
                        qs = -1
                    log_debug(
                        f"[frame] nf={noise_floor_db:.1f} gate={gate_db:.1f} lvl={level_db:.1f} voiced={int(voiced)} in={int(in_speech)} qa={qa} qs={qs}")
                    last_telemetry = now

        except KeyboardInterrupt:
            pass
        finally:
            stop_evt.set()
            worker.join(timeout=2.0)


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        main()
    except KeyboardInterrupt:
        log_info("\nStopped.")
        sys.exit(0)
