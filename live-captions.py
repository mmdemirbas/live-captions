#!/usr/bin/env python3
import argparse, queue, sys, time, sounddevice as sd
from faster_whisper import WhisperModel
import webrtcvad, numpy as np
import scipy.signal


def int16_to_float32(x): return x.astype(np.float32) / 32768.0


mic_device_name = "MacBook Pro Microphone"
system_device_name = "BlackHole 2ch"


# TODO: List only audio devices? Does it work with speakers?
# TODO: --source: Support specifying a file as input?
# TODO: --target: Improve output representation (e.g., file output (SRT etc.), screen overlay, etc.)
# TODO: Improve output details (e.g., timestamps, confidence scores, etc.)

def main():
    # ── PARSE CLI ARGUMENTS ───────────────────────────────────────────────────────────────────────
    p = argparse.ArgumentParser()

    # --source specifies the audio source device
    p.add_argument("--source", type=str, default=None,
                   help="mic=default input, system=BlackHole loop-back, or explicit PortAudio index number")

    # --model specifies the Whisper model to use
    p.add_argument("--model", default="small.en", help="Whisper size or path")

    # --latency specifies how far behind the live stream to process
    p.add_argument("--latency", type=float, default=0.8, help="max seconds behind live")

    args = p.parse_args()

    # ── FIND AUDIO DEVICES ────────────────────────────────────────────────────────────────────────

    mbp_mic_index = next(
        (i for i, d in enumerate(sd.query_devices()) if mic_device_name in d["name"]),
        None)

    blackhole_index = next(
        (i for i, d in enumerate(sd.query_devices()) if system_device_name in d["name"]), None)

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
            print(f"     mic : ⚠️ '{mic_device_name}' not found")
        else:
            print(f"     mic : MBP microphone ({mbp_mic_index})")

        if blackhole_index is None:
            print(
                f"  system : ⚠️ You need BlackHole to listen system audio – `brew install blackhole-2ch` & reboot")
        else:
            print(f"  system : BlackHole loop-back ({blackhole_index})")

        sd.query_devices()
        sys.exit(1)

    # Print the device name
    device_info = sd.query_devices(device_index)
    print(f"Using device: {device_info['name']} (index {device_index})")

    # ── INIT AUDIO MODEL ──────────────────────────────────────────────────────────────────────────

    # Remove the VAD error and make the code sample-rate-agnostic
    device_sr = int(sd.query_devices(device_index)["default_samplerate"])
    frame_bytes = 960  # 30 ms at 16 kHz
    samplerate, block = 16000, frame_bytes  # ASR works at 16 kHz

    model = WhisperModel(args.model, compute_type="int8")
    q = queue.Queue()
    vad = webrtcvad.Vad(2)

    def callback(indata, _frames, _time, status):
        q.put(bytes(indata))

    with sd.RawInputStream(
            dtype="int16", channels=1, samplerate=samplerate,
            blocksize=block, callback=callback, device=device_index):

        buf, last_print = b"", time.time()
        while True:
            # Read audio data from the queue
            buf += q.get()
            if len(buf) < samplerate * args.latency * 2:  # 16 kHz * sec * 2 B
                continue

            # If the first 30 ms is silence, wait for speech
            if not vad.is_speech(buf[:frame_bytes], samplerate):
                continue

            # Auto-resample for 48 kHz devices
            raw = np.frombuffer(buf, np.int16)
            if device_sr != 16000:
                # simple linear resample – good enough for speech
                raw = scipy.signal.resample_poly(raw, 16000, device_sr)
            array = raw.astype(np.int16)

            # Process the audio buffer
            segments, _ = model.transcribe(int16_to_float32(array), beam_size=1)
            for seg in segments: print(seg.text, flush=True)

            # Reset after commit
            buf = b""

            # Keep prompt responsive
            if time.time() - last_print > 5: print("", flush=True); last_print = time.time()


if __name__ == "__main__": main()
