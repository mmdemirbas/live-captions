#!/usr/bin/env python3
import argparse, queue, sys, time, sounddevice as sd
from faster_whisper import WhisperModel
import webrtcvad, numpy as np
import signal
import scipy
from collections import deque
from log import log_info, log_warn, log_error, log_debug

mic_device_name = "MacBook Pro Microphone"
sys_device_name = "BlackHole 2ch"


def main():
    # ── PARSE CLI ARGUMENTS ───────────────────────────────────────────────────────────────────────

    default_model = "medium.en"
    default_end_silence_ms = 260  # good range: 180–220 ms. Below ~150 ms often cut words. Above 250 ms feel less snappy.
    default_max_segment_ms = 1000  # ms - force to finalize the segment if exceeds this duration
    default_noise_gate_db = -52.0  # dBFS
    default_speech_threshold = 0  # 0=keeps more, 3=filters more
    default_beam_size = 3  # 1=fastest, 10=most accurate (higher is slower and more accurate)

    p = argparse.ArgumentParser(description="Live captions from mic or system audio.")
    p.add_argument("--source", type=str, help="mic | sys | device index")

    # Preset profiles for quick tuning
    p.add_argument("--profile", choices=["fast", "default", "slow"],
                   help="Tune latency vs accuracy")

    # Granular tuning parameters
    p.add_argument("--model", default=default_model, help="Whisper size/path")
    p.add_argument("--end-silence-ms", type=int, default=default_end_silence_ms,
                   help="Finalize segment after this many ms of silence (ms)")
    p.add_argument("--max-segment-ms", type=int, default=default_max_segment_ms,
                   help="Force finalize if a segment exceeds this duration (ms)")
    p.add_argument("--noise-gate-db", type=float, default=default_noise_gate_db,
                   help="Frames quieter than this dBFS are treated as non-speech (e.g., -45)")
    p.add_argument("--speech-threshold", type=int, default=default_speech_threshold,
                   choices=[0, 1, 2, 3],
                   help="3=filters more non-speech but may miss quiet speech")
    p.add_argument("--beam-size", type=int, default=default_beam_size,
                   help="Beam size for Whisper decoding. 1-2 fast, 3 balanced, 5+ slower/more accurate")

    # Debug parameters
    p.add_argument("--timestamps", action="store_true",
                   help="Show wall-clock timestamp prefix for debugging")

    args = p.parse_args()

    # ── APPLY PROFILE  ────────────────────────────────────────────────────────────────────────────

    if args.profile == "fast":
        model = "base.en"
        end_silence_ms = 180
        max_segment_ms = 500
        noise_gate_db = -42.0
        speech_threshold = 2
        beam_size = 1

    elif args.profile == "default" or args.profile is None:
        model = default_model
        end_silence_ms = default_end_silence_ms
        max_segment_ms = default_max_segment_ms
        noise_gate_db = default_noise_gate_db
        speech_threshold = default_speech_threshold
        beam_size = default_beam_size

    elif args.profile == "slow":
        model = "large-v2"
        end_silence_ms = 300
        max_segment_ms = 5000
        noise_gate_db = -55.0
        speech_threshold = 0
        beam_size = 8

    else:
        log_error(f"Unknown profile '{args.profile}'")
        sys.exit(1)

    # Overwrite defaults if specified
    args.model = model if args.model is None or args.model == default_model else args.model
    args.end_silence_ms = end_silence_ms if args.end_silence_ms is None or args.end_silence_ms == default_end_silence_ms else args.end_silence_ms
    args.max_segment_ms = max_segment_ms if args.max_segment_ms is None or args.max_segment_ms == default_max_segment_ms else args.max_segment_ms
    args.noise_gate_db = noise_gate_db if args.noise_gate_db is None or args.noise_gate_db == default_noise_gate_db else args.noise_gate_db
    args.speech_threshold = speech_threshold if args.speech_threshold is None or args.speech_threshold == default_speech_threshold else args.speech_threshold
    args.beam_size = beam_size if args.beam_size is None or args.beam_size == default_beam_size else args.beam_size

    # ── FIND AUDIO DEVICE ─────────────────────────────────────────────────────────────────────────

    def list_input_devices():
        devices = sd.query_devices()
        result = []
        for idx, d in enumerate(devices):
            max_in = d.get("max_input_channels", 0)
            name = d.get("name", f"Device {idx}")
            if max_in and max_in > 0:
                result.append({"index": idx, "name": name})
        return result

    input_devices = list_input_devices()

    mbp_mic_index = next((d["index"] for d in input_devices if mic_device_name in d["name"]), None)
    blackhole_index = next((d["index"] for d in input_devices if sys_device_name in d["name"]),
                           None)

    if args.source is None:
        device_index = None
    elif args.source == "mic":
        device_index = mbp_mic_index
    elif args.source == "sys":
        device_index = blackhole_index
    else:
        try:
            device_index = int(args.source)
        except ValueError:
            device_index = None

    # If no valid device is specified, print available devices and exit
    if device_index is None:
        log_error("Please specify a valid --source:\n")

        if mbp_mic_index is None:
            log_warn(f"mic : ⚠️ '{mic_device_name}' not found")
        else:
            log_info(f"mic : {mic_device_name} ({mbp_mic_index})")

        if blackhole_index is None:
            log_warn(
                f"sys : ⚠️ '{sys_device_name}' not found. Follow README.")
        else:
            log_info(f"sys : {sys_device_name} ({blackhole_index})")

        for d in input_devices:
            log_info(f"{d['index']:>3} : {d['name']}")
        sys.exit(1)

    # Print parameters
    log_info(f"Live captions")
    log_info(f"--source           : {args.source}")
    log_info(f"--profile          : {args.profile or 'default'}")
    log_info(f"--model            : {args.model}")
    log_info(f"--end-silence-ms   : {args.end_silence_ms} ms")
    log_info(f"--max-segment-ms   : {args.max_segment_ms} ms")
    log_info(f"--noise-gate-db    : {args.noise_gate_db} dBFS")
    log_info(f"--speech-threshold : {args.speech_threshold}")
    log_info(f"--beam-size        : {args.beam_size}")
    log_info(f"--timestamps       : {'on' if args.timestamps else 'off'}\n")

    # ── INIT AUDIO MODEL ──────────────────────────────────────────────────────────────────────────

    device_sr = int(sd.query_devices(device_index)["default_samplerate"])  # e.g. 48000
    target_sr = 16000  # Whisper/VAD rate

    frame_ms = 30
    samples_per_frame_dev = int(device_sr * frame_ms / 1000)
    samples_per_frame_16k = int(target_sr * frame_ms / 1000)  # 480 at 16 kHz
    bytes_per_sample = 2  # int16
    frame_bytes_dev = samples_per_frame_dev * bytes_per_sample

    # Caps to protect memory and latency
    max_buf_ms = 1000  # keep at most 1000ms of unread device bytes
    max_buf_bytes = int(device_sr * max_buf_ms / 1000) * bytes_per_sample

    # VAD operates at 16 kHz on 10/20/30 ms frames. We use 30 ms.
    vad = webrtcvad.Vad(args.speech_threshold)

    # Model
    # Prefer auto selection; will use float16 on GPU if available, int8/int16 on CPU when supported
    model = WhisperModel(args.model, compute_type="auto")

    # Buffers and state
    q = queue.Queue(maxsize=64)  # bounded queue to avoid unbounded growth
    buf = bytearray()  # device-rate raw bytes
    in_speech = False
    silence_frames = 0
    segment_len_frames = 0
    consec_voiced_to_enter = 1  # require 2 voiced frames to start
    enter_voiced_streak = 0

    # Keep a small lookback/tail buffer (200 ms) in 16 kHz for context at segment start
    lookback_ms = 200
    lookback_frames = lookback_ms // frame_ms
    lookback_16k = deque(maxlen=lookback_frames)

    # Keep a small tail padding when segment ends
    tail_ms = 200
    tail_frames = tail_ms // frame_ms

    # Accumulate voiced segment in 16 kHz float32
    current_segment_16k = []

    dropped_frames = 0
    last_warn = 0.0

    def resample_16k_int16(frame_bytes_dev: bytes) -> np.ndarray:
        # Convert device-rate bytes -> float32 -> resample -> float32 in [-1, 1]
        x = np.frombuffer(frame_bytes_dev, dtype=np.int16).astype(
            np.float32) / 32768.0  # int16 to float32
        if device_sr == target_sr:
            yf = x
        else:
            yf = scipy.signal.resample_poly(x, target_sr, device_sr).astype(np.float32)
        # Ensure an exact 30 ms at 16 kHz (guard against rounding)
        if len(yf) != samples_per_frame_16k:
            if len(yf) > samples_per_frame_16k:
                yf = yf[:samples_per_frame_16k]
            else:
                yf = np.pad(yf, (0, samples_per_frame_16k - len(yf)))
        return yf

    def rms_dbfs(yf: np.ndarray) -> float:
        eps = 1e-12
        rms = np.sqrt(np.mean(yf * yf) + eps)
        return 20.0 * np.log10(rms + eps)

    def simple_noise_reduce(yf, sr=16000):
        return scipy.signal.lfilter([1, -0.97], [1], yf)  # high-pass at ~100 Hz

    def vad_majority_3x10ms(yf: np.ndarray) -> bool:
        # Split 30 ms (480 samples) into 3 chunks of 10 ms (160 samples)
        chunk = samples_per_frame_16k // 3  # 160
        votes = 0
        for i in range(3):
            sub = yf[i * chunk:(i + 1) * chunk]
            y_i16 = (np.clip(sub, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()
            if vad.is_speech(y_i16, target_sr):
                votes += 1
        return votes >= 2  # majority

    def is_voiced_30ms_16k(yf: np.ndarray, gate_db: float) -> bool:
        # Require both VAD speech and energy above gate
        yf_reduced = simple_noise_reduce(yf)
        voiced = vad_majority_3x10ms(yf_reduced)
        level_db = rms_dbfs(yf_reduced)
        return voiced and (level_db >= gate_db)

    def callback(indata, _frames, _time, status):
        nonlocal dropped_frames, last_warn
        try:
            if status:
                now = time.time()
                # Throttle status warnings to once per 5s
                if now - last_warn > 5:
                    log_warn(f"[audio] status: {status}")
                    last_warn = now
            # If the queue is full, drop the oldest frame to keep latency bounded
            if q.full():
                try:
                    _ = q.get_nowait()
                    dropped_frames += 1
                except queue.Empty:
                    pass
            q.put_nowait(bytes(indata))
        except Exception as e:
            now = time.time()
            if now - last_warn > 5:
                log_warn(f"[audio] callback error: {e}")
                last_warn = now

    # Print the device name
    device_info = sd.query_devices(device_index)
    log_warn(f"Using device: {device_info['name']} ({device_index})")

    try:
        stream = sd.RawInputStream(
            dtype="int16", channels=1, samplerate=device_sr,
            blocksize=samples_per_frame_dev,  # 30 ms frames
            callback=callback, device=device_index)
    except Exception as e:
        log_error(f"Failed to open audio stream: {e}")
        sys.exit(2)

    with stream:
        log_warn("Listening... Press Ctrl+C to stop.")
        last_stats = time.time()

        while True:
            try:
                # Read one frame; use a timeout to allow housekeeping and clean exit
                try:
                    chunk = q.get(timeout=1.0)
                except queue.Empty:
                    # No audio arrived for 1s; continue and allow Ctrl+C handling
                    continue

                buf.extend(chunk)

                # Ensure buf does not exceed cap (keep most recent tail)
                if len(buf) > max_buf_bytes:
                    # Drop oldest bytes, keep latest max_buf_bytes
                    buf[:] = buf[-max_buf_bytes:]

                # Process exactly one 30 ms frame when available
                if len(buf) < frame_bytes_dev:
                    continue

                frame_dev = bytes(buf[:frame_bytes_dev])
                del buf[:frame_bytes_dev]

                # Resample this frame to 16 kHz for VAD and model
                frame_16k = resample_16k_int16(frame_dev)
                voiced = is_voiced_30ms_16k(frame_16k, args.noise_gate_db)

                if not in_speech:
                    # Keep rolling lookback
                    lookback_16k.append(frame_16k)
                    if voiced:
                        # Enter speech: prepend lookback for breathing room/context
                        enter_voiced_streak += 1
                        if enter_voiced_streak >= consec_voiced_to_enter:
                            in_speech = True
                            silence_frames = 0
                            segment_len_frames = 0
                            current_segment_16k = list(lookback_16k)  # shallow copy of frames
                    else:
                        enter_voiced_streak = 0
                        # otherwise remain idle
                else:
                    segment_len_frames += 1
                    if voiced:
                        current_segment_16k.append(frame_16k)
                        silence_frames = 0
                        trailing_tail_needed = tail_frames
                    else:
                        silence_frames += 1
                        # Add a few silent frames as tail padding (to avoid cutting consonants)
                        if 'trailing_tail_needed' in locals() and trailing_tail_needed > 0:
                            current_segment_16k.append(frame_16k)
                            trailing_tail_needed -= 1

                    # End on silence OR force flush on long segments
                    end_on_silence = (silence_frames * frame_ms) >= args.end_silence_ms
                    force_on_timeout = (segment_len_frames * frame_ms) >= args.max_segment_ms
                    if end_on_silence or force_on_timeout:
                        # Finalize segment
                        if current_segment_16k:
                            seg = np.concatenate(current_segment_16k, axis=0).astype(np.float32)
                            try:
                                # Transcribe finalized segment
                                # For better accuracy: small beam; temperature 0 for determinism
                                segments, _ = model.transcribe(
                                    seg,
                                    beam_size=args.beam_size,
                                    temperature=0.1,
                                    vad_filter=False,
                                    word_timestamps=False,
                                    language="en"
                                )
                                if args.timestamps:
                                    now_ts = time.strftime("%H:%M:%S")
                                    for s in segments:
                                        print(f"[{now_ts}] {s.text}", flush=True)
                                else:
                                    for s in segments:
                                        print(s.text, flush=True)
                            except KeyboardInterrupt:
                                # Gracefully abort an in-flight decode
                                log_warn("\nStopping...")
                                return
                            except Exception as e:
                                # Log and continue; don't crash the loop
                                log_error(f"[stt] transcribe error: {e}")

                        # Reset state
                        in_speech = False
                        silence_frames = 0
                        segment_len_frames = 0
                        enter_voiced_streak = 0
                        current_segment_16k = []
                        lookback_16k.clear()

                # Periodic lightweight diagnostics
                now = time.time()
                if now - last_stats > 10:
                    if dropped_frames:
                        log_debug(f"dropped frames in last 10s: {dropped_frames}")
                        dropped_frames = 0
                    last_stats = now

            except KeyboardInterrupt:
                log_warn("\nStopping...")
                return


if __name__ == "__main__":
    try:
        # Optional: ignore SIGINT default handler so we only handle it here
        signal.signal(signal.SIGINT, signal.default_int_handler)
        main()
    except KeyboardInterrupt:
        # Final catch-all to avoid stack traces from deep inside libraries
        log_info("\nStopped.")
        sys.exit(0)
