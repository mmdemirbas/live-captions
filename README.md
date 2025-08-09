# Live Captions

<!-- TOC -->
* [Live Captions](#live-captions)
  * [Prerequisites](#prerequisites)
    * [Setup Blackhole](#setup-blackhole)
  * [How does it work?](#how-does-it-work)
    * [Device selection](#device-selection)
    * [Audio capture](#audio-capture)
    * [Framing and resampling](#framing-and-resampling)
    * [VAD (Voice Activity Detection)](#vad-voice-activity-detection)
    * [Segmentation (a small state machine)](#segmentation-a-small-state-machine)
    * [Transcription](#transcription)
    * [Loop](#loop)
  * [Core Concepts](#core-concepts)
    * [What is VAD?](#what-is-vad)
    * [Why 16 kHz? Doesn’t it hurt quality?](#why-16-khz-doesnt-it-hurt-quality)
    * [What is "segmentation strategy"?](#what-is-segmentation-strategy)
    * [What is beam and beam_size](#what-is-beam-and-beam_size)
    * [What are the options for --model and their differences?](#what-are-the-options-for---model-and-their-differences)
  * [Possible Use-Cases](#possible-use-cases)
    * [Overlay subtitles on videos](#overlay-subtitles-on-videos)
<!-- TOC -->

Speech-to-text transcription in real-time, using a microphone or system audio loopback.

## Prerequisites

- Python 3.10+
- A microphone (internal or external) or a loopback device like BlackHole to capture system audio.
- Internet connection for downloading dependencies and models.

### Setup Blackhole

1. Install with `brew install blackhole-2ch`
2. Reboot.
3. Run `open -a "Audio MIDI Setup"`
4. Add → Multi-Output Device
5. Tick Built-in Output and BlackHole 2-ch
6. Make this Multi-Output the system output → you still hear audio, BlackHole gets the same signal.
7. Use `--source system` in the script to capture system audio.

Alternatively, you can use `--source <index>` to specify a device index directly.

## How does it work?

### Device selection

Lists audio devices and chooses either the MacBook microphone, a loopback like BlackHole (system
audio), or an explicit index.

### Audio capture

Opens a 1‑channel RawInputStream delivering 30 ms frames of 16‑bit samples at the device’s native
sample rate (often 48 kHz).

### Framing and resampling

Each 30 ms frame is resampled to 16 kHz. That’s the sample rate expected by the VAD and Whisper.

### VAD (Voice Activity Detection)

For each 30 ms 16 kHz frame, the WebRTC VAD decides voiced vs. non-voiced (speech vs. not).

### Segmentation (a small state machine)

- Maintain a small lookback (e.g., 200 ms) so the start of a segment includes a bit of context from
  just before speech begins.
- While in speech: keep appending voiced frames; track silence.
- When silence exceeds a threshold (e.g., 300 ms), finalize the segment, add a short tail, then send
  the concatenated audio to Whisper.

### Transcription

Run faster-whisper to produce text for the finalized chunk. Print the text. Optionally include
timestamps for debugging.

### Loop

Repeat indefinitely, handling frame-by-frame.

## Core Concepts

### What is VAD?

Voice Activity Detection. It classifies short audio frames as speech or non-speech (silence, noise,
music). Using VAD to group speech into “segments” reduces false starts and improves accuracy,
because the ASR model sees coherent chunks rather than arbitrary slices.

### Why 16 kHz? Doesn’t it hurt quality?

Whisper (and many ASR systems) are trained on 16 kHz speech. Human speech intelligibility is mostly
below ~8 kHz, and 16 kHz captures that band with margin. Downsampling from 48 kHz to 16kHz does not
harm recognition accuracy; it can actually help by matching the model’s training conditions and
reducing noise/out-of-band content.

### What is "segmentation strategy"?

“Segmentation strategy” means how audio portions are selected. It’s the logic for deciding:

- When to start a speech segment (enter speech).
- How much pre-roll context to include (lookback).
- When to stop (after X ms of silence).
- Whether to add padding at ends (tail).

Good segmentation improves accuracy and naturalness; poor segmentation cuts words and confuses the
model.

### What is beam and beam_size

Beam search is a decoding method that keeps several top candidate transcriptions as it processes
audio, not just the single best at every step. beam_size is how many hypotheses are kept. Larger
beam_size can yield better accuracy (especially in tricky audio) but is slower. Typical ranges: 1–5
for real-time; up to ~10 for more accuracy.

### What are the options for --model and their differences?

faster-whisper supports the Whisper family: tiny, base, small, medium, large-v2 (and “.en” variants
optimized for English only).

- tiny/tiny.en: fastest, least accurate, tiny memory footprint.
- base/base.en: fast, slightly better accuracy.
- small/small.en: balanced speed/accuracy.
- medium/medium.en: slower, more accurate (your current default).
- large-v2: highest accuracy, slowest and most memory hungry.

“.en” variants typically perform a bit better on English and are smaller/faster, but they won’t
transcribe other languages.

## Possible Use-Cases

### Overlay subtitles on videos

You can overlay subtitle with mpv:

    ./live-captions.py --source system --model small.en > /tmp/live.srt &
    mpv --sub-file=/tmp/live.srt /path/to/Battlestar_Galactica.mkv

mpv auto-reloads the SRT every few seconds; captions appear on-screen.
