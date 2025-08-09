#!/usr/bin/env python3
import argparse, queue, sys, time, sounddevice as sd
from faster_whisper import WhisperModel
import webrtcvad, numpy as np
import signal
import scipy
from collections import deque

mic_device_name = "MacBook Pro Microphone"
system_device_name = "BlackHole 2ch"


# TODO: Add a good README doc (what is this, prerequisities etc)
# TODO: Add preset profiles
# TODO: --source: Support specifying a file as input?
# TODO: --target: Improve output representation (e.g., file output (SRT etc.), screen overlay, etc.)
# TODO: Improve output details (e.g., timestamps, confidence scores, etc.)
# TODO: Minimize resampling cost: If the device can be opened at 16 kHz, do so and skip resample.

def main():
    # ── PARSE CLI ARGUMENTS ───────────────────────────────────────────────────────────────────────
    default_model = "medium.en"
    default_speech_threshold = 2
    default_end_silence = 300

    p = argparse.ArgumentParser(description="Live captions from mic or system audio.")
    p.add_argument("--source", type=str, help="mic | system | device index")
    p.add_argument("--profile", choices=["realtime", "optimized", "default", "accurate"],
                   help="Tune latency vs accuracy")
    p.add_argument("--model", default=default_model, help="Whisper size/path")
    p.add_argument("--speech-threshold", type=int, default=default_speech_threshold,
                   choices=[0, 1, 2, 3],
                   help="Higher=more strict. If quiet speech is missed, lower it.")
    p.add_argument("--end-silence", type=int, default=default_end_silence,
                   help="Milliseconds of silence to finalize a line/segment")
    p.add_argument("--timestamps", action="store_true",
                   help="Show wall-clock timestamp prefix for debugging")
    args = p.parse_args()

    # ── APPLY PROFILE  ────────────────────────────────────────────────────────────────────────────

    if args.profile == "realtime":
        model = "small.en"
        speech_threshold = 1  # 1-2
        end_silence = 200  # 200-250
        beam_size = 1  # 1-2

    elif args.profile == "optimized":
        model = "medium.en"
        speech_threshold = 2
        end_silence = 200  # 150–250 ms is a good range; below ~150 ms you risk cutting words
        beam_size = 4  # 1-3 for real-time; 5+ for harder audio, but it’ll add delay
        frame_size = 20  # 20 ms (more frequent decisions) if CPU allows. 10 ms increases overhead
        tail_padding = 100  # 80–120 ms helps avoid chopping final consonants

    elif args.profile == "default" or args.profile is None:
        model = default_model
        speech_threshold = default_speech_threshold
        end_silence = default_end_silence
        beam_size = 5  # Default beam size for Whisper

    elif args.profile == "accurate":
        model = "large-v3"
        speech_threshold = 2
        end_silence = 400  # 300-400
        beam_size = 10  # 5-10

    else:
        print(f"Unknown profile '{args.profile}'")
        sys.exit(1)

    # Overwrite defaults if specified
    args.model = model if args.model is None or args.model == default_model else args.model
    args.speech_threshold = speech_threshold if args.speech_threshold is None or args.speech_threshold == default_speech_threshold else args.speech_threshold
    args.end_silence = end_silence if args.end_silence is None or args.end_silence == default_end_silence else args.end_silence

    # Print parameters
    print(f"Live captions")
    print(f"--source           : {args.source}")
    print(f"--profile          : {args.profile or 'default'}")
    print(f"--model            : {args.model}")
    print(f"--speech-threshold : {args.speech_threshold}")
    print(f"--end-silence      : {args.end_silence} ms")
    print(f"--timestamps       : {'on' if args.timestamps else 'off'}")
    print(f"  beam_size        : {beam_size}")
    print("")

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
    blackhole_index = next((d["index"] for d in input_devices if system_device_name in d["name"]),
                           None)

    if args.source is None:
        device_index = None
    elif args.source == "mic":
        device_index = mbp_mic_index
    elif args.source == "system":
        device_index = blackhole_index
    else:
        try:
            device_index = int(args.source)
        except ValueError:
            device_index = None

    # If no valid device is specified, print available devices and exit
    if device_index is None:
        print("Please specify a valid --source:")

        if mbp_mic_index is None:
            print(f"   mic : ⚠️ '{mic_device_name}' not found")
        else:
            print(f"   mic : MBP microphone ({mbp_mic_index})")

        if blackhole_index is None:
            print(
                f"system : ⚠️ You need BlackHole to listen system audio – Follow README")
        else:
            print(f"system : BlackHole loop-back ({blackhole_index})")

        for d in input_devices:
            if d["index"] not in (mbp_mic_index, blackhole_index):
                print(f"{d['index']:>6} : {d['name']}")
        sys.exit(1)

    # Print the device name
    device_info = sd.query_devices(device_index)
    print(f"Using device: {device_info['name']} (index {device_index})")

    # ── INIT AUDIO MODEL ──────────────────────────────────────────────────────────────────────────

    device_sr = int(sd.query_devices(device_index)["default_samplerate"])  # e.g. 48000
    target_sr = 16000  # Whisper’s native rate

    frame_ms = 30
    samples_per_frame_dev = int(device_sr * frame_ms / 1000)
    samples_per_frame_16k = int(target_sr * frame_ms / 1000)
    bytes_per_sample = 2  # int16
    frame_bytes_dev = samples_per_frame_dev * bytes_per_sample

    # VAD operates at 16 kHz on 10/20/30 ms frames. We use 30 ms.
    vad = webrtcvad.Vad(args.speech_threshold)

    # Model
    # Prefer auto selection; will use float16 on GPU if available, int8/int16 on CPU when supported
    model = WhisperModel(args.model, compute_type="auto")

    # Buffers and state
    q = queue.Queue()
    buf = bytearray()  # device-rate raw bytes
    in_speech = False
    silence_frames = 0

    # Keep a small look-back buffer (200 ms) in 16 kHz for context at segment start
    lookback_ms = 200
    lookback_frames = lookback_ms // frame_ms
    lookback_16k = deque(maxlen=lookback_frames)

    # Keep a small tail padding when segment ends
    tail_ms = 200
    tail_frames = tail_ms // frame_ms

    # Accumulate voiced segment in 16 kHz float32
    current_segment_16k = []

    def resample_16k_int16(frame_bytes_dev: bytes) -> np.ndarray:
        # Convert device-rate bytes -> float32 -> resample -> float32 in [-1, 1]
        x = np.frombuffer(frame_bytes_dev, dtype=np.int16)
        xf = x.astype(np.float32) / 32768.0  # int16 to float32
        if device_sr == target_sr:
            yf = xf
        else:
            yf = scipy.signal.resample_poly(xf, target_sr, device_sr).astype(np.float32)
        # Ensure exact 30 ms at 16 kHz (guard against rounding)
        if len(yf) != samples_per_frame_16k:
            if len(yf) > samples_per_frame_16k:
                yf = yf[:samples_per_frame_16k]
            else:
                pad = np.zeros(samples_per_frame_16k - len(yf), dtype=np.float32)
                yf = np.concatenate([yf, pad])
        return yf

    def is_voiced_30ms_16k(yf: np.ndarray) -> bool:
        # VAD expects 16-bit PCM bytes at 16 kHz, 30 ms
        y_int16 = (np.clip(yf, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()
        return vad.is_speech(y_int16, target_sr)

    def callback(indata, _frames, _time, status):
        if status:
            # Dropouts or overflows can harm accuracy; print once every few seconds if needed
            pass
        q.put(bytes(indata))

    with sd.RawInputStream(
            dtype="int16", channels=1, samplerate=device_sr,
            blocksize=samples_per_frame_dev,  # 30 ms frames
            callback=callback, device=device_index):

        last_print = time.time()
        trailing_tail_needed = 0

        print("Listening... Press Ctrl+C to stop.")
        while True:
            try:
                # Gather exactly one 30 ms frame worth of bytes
                while len(buf) < frame_bytes_dev:
                    buf.extend(q.get())

                frame_dev = bytes(buf[:frame_bytes_dev])
                del buf[:frame_bytes_dev]

                # Resample this frame to 16 kHz for VAD and model
                frame_16k = resample_16k_int16(frame_dev)
                voiced = is_voiced_30ms_16k(frame_16k)

                if not in_speech:
                    # Keep rolling lookback
                    lookback_16k.append(frame_16k)
                    if voiced:
                        # Enter speech: prepend lookback for breathing room/context
                        in_speech = True
                        silence_frames = 0
                        current_segment_16k = list(lookback_16k)  # shallow copy of frames
                        # otherwise remain idle
                else:
                    if voiced:
                        current_segment_16k.append(frame_16k)
                        silence_frames = 0
                        trailing_tail_needed = tail_frames
                    else:
                        silence_frames += 1
                        # Add a few silent frames as tail padding (to avoid cutting consonants)
                        if trailing_tail_needed > 0:
                            current_segment_16k.append(frame_16k)
                            trailing_tail_needed -= 1

                        if silence_frames * frame_ms >= args.end_silence:
                            # Finalize segment
                            if current_segment_16k:
                                seg = np.concatenate(current_segment_16k, axis=0).astype(np.float32)
                                try:
                                    # Transcribe finalized segment
                                    # For better accuracy: small beam; temperature 0 for determinism
                                    segments, _ = model.transcribe(
                                        seg,
                                        beam_size=beam_size,
                                        temperature=0.0,
                                        vad_filter=False,
                                        word_timestamps=False,
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
                                    print("\nStopping...", flush=True)
                                    return

                            # Reset state
                            in_speech = False
                            silence_frames = 0
                            current_segment_16k = []
                            lookback_16k.clear()

                # Keep prompt responsive (flush heartbeat)
                if time.time() - last_print > 5:
                    print("", flush=True)
                    last_print = time.time()

            except KeyboardInterrupt:
                print("\nStopping...", flush=True)
                return


if __name__ == "__main__":
    try:
        # Optional: ignore SIGINT default handler so we only handle it here
        signal.signal(signal.SIGINT, signal.default_int_handler)
        main()
    except KeyboardInterrupt:
        # Final catch-all to avoid stack traces from deep inside libraries
        print("\nStopped.", flush=True)
        sys.exit(0)
