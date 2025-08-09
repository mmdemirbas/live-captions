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
  * [TODO](#todo)
    * [Consider named pipes (FIFO)](#consider-named-pipes-fifo)
    * [Incremental correction](#incremental-correction)
      * [How to correct past words](#how-to-correct-past-words)
      * [Algorithm sketch](#algorithm-sketch)
      * [Trade-offs:](#trade-offs)
      * [Implementation notes:](#implementation-notes)
    * [Overlay subtitles on screen](#overlay-subtitles-on-screen)
    * [Other speed & accuracy optimizations](#other-speed--accuracy-optimizations)
      * [Denoising and AGC](#denoising-and-agc)
      * [Energy gating prior to VAD](#energy-gating-prior-to-vad)
      * [Hotwords/lexicons](#hotwordslexicons)
      * [Device/sample-rate](#devicesample-rate)
      * [Batch inference and concurrency](#batch-inference-and-concurrency)
    * [Future modularization](#future-modularization)
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

## TODO

### Consider named pipes (FIFO)

- A named pipe (FIFO) is a special file that streams data between processes.
    - You create one with mkfifo /tmp/live.srt.
    - Your script writes SRT lines into it as if it were a file.
    - A player can read from it like a file path, effectively “following” new subtitles as they’re
      written.
- Players like mpv can reload or continually read subtitles from a FIFO, providing near-live overlay
  without writing to disk repeatedly.

### Incremental correction

#### How to correct past words

Implement online, incremental decoding with revision:

- Keep a rolling context window (e.g., last 10–15 seconds).
- Every N milliseconds, re-decode that window with the latest audio appended.
- Align the new hypothesis with the previous one (using timestamps or token alignment).
- Commit only the stable prefix: the part that hasn’t changed across, say, two consecutive decodes.
- Replace the tail (unstable suffix) on screen with the revised text.

#### Algorithm sketch

Maintain a ring buffer of 10–15 s audio.

Every 300–500 ms:

- Decode buffer with overlap (e.g., last 2–3 s overlaps previous decode).
- Compare previous and current token sequences or word timestamps.
- Find the longest common prefix beyond a stability window.
- Emit new text for the unstable region only.

#### Trade-offs:

- More compute (frequent redecoding).
- Lower perceived latency and higher perceived accuracy because words get refined quickly.

#### Implementation notes:

- faster-whisper exposes segment and optional word timestamps to support alignment.
- Will need an “emitter” that manages a “committed text” buffer and a “live tail” that’s allowed to
  change.

### Overlay subtitles on screen

- Options on macOS:
    - PyQt/PySide: a frameless, always-on-top, click-through transparent window showing text.
    - Tkinter: possible but less polished.
    - PyObjC (AppKit): native control over transparency, window level (floating above full-screen),
      font rendering, and click-through.
    - Or render into a player (mpv) via named pipe SRT (already workable).

- Minimal PyQt idea:
    - Create a QMainWindow with Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint and transparent
      background (Qt.WA_TranslucentBackground).
    - Render large antialiased text with outline/shadow for readability.
    - Optionally ignore mouse events so it doesn’t block clicks.

### Other speed & accuracy optimizations

#### Denoising and AGC

Light noise suppression and automatic gain control improve VAD and ASR. WebRTC has NS/AGC modules,
or use simple filters. Beware over-aggressive NS that warps speech.

#### Energy gating prior to VAD

Quick RMS threshold to skip obvious silence reduces VAD and decode workload.

#### Hotwords/lexicons

Whisper doesn’t natively use external dictionaries, but you can bias results using initial_prompt
and condition_on_previous_text. This can stabilize terminology.

#### Device/sample-rate

Open the device at 16 kHz if possible to avoid resampling cost.

#### Batch inference and concurrency

If you do incremental redecoding, overlap I/O and compute with threads/async to hide compute
latency.

### Future modularization

When the main script exceeds ~300 lines and mixes concerns. A common split:

- audio_io.py: device discovery, capture, resampling.
- vad_segmenter.py: frame logic, state machine.
- transcriber.py: model init and decode helpers.
- cli.py: argument parsing and presets.
- outputs/srt_writer.py: writers/formatters.
- ui/overlay.py: on-screen overlay code.
- main.py: glues components.
