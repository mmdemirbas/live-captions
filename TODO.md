# TODO

<!-- TOC -->
* [TODO](#todo)
    * [Overlay subtitles on screen](#overlay-subtitles-on-screen)
    * [Support "--source file" option](#support---source-file-option)
    * [Improve output representation](#improve-output-representation)
      * [Named pipes (FIFO)](#named-pipes-fifo)
    * [Performance](#performance)
      * [Minimize resampling cost](#minimize-resampling-cost)
    * [Speed & accuracy optimizations](#speed--accuracy-optimizations)
      * [Denoising and AGC](#denoising-and-agc)
      * [Energy gating prior to VAD](#energy-gating-prior-to-vad)
      * [Hotwords/lexicons](#hotwordslexicons)
      * [Device/sample-rate](#devicesample-rate)
      * [Batch inference and concurrency](#batch-inference-and-concurrency)
    * [Incremental correction](#incremental-correction)
      * [How to correct past words](#how-to-correct-past-words)
      * [Algorithm sketch](#algorithm-sketch)
      * [Trade-offs:](#trade-offs)
      * [Implementation notes:](#implementation-notes)
    * [Modularization](#modularization)
<!-- TOC -->

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

### Support "--source file" option

Add support for reading from a file, allowing pre-recorded audio to be processed.

### Improve output representation

- Improve output representation (e.g., file output (SRT etc.), screen overlay, etc.)
- Consider introducing --target option

#### Named pipes (FIFO)

A named pipe (FIFO) is a special file that streams data
between processes.

- You create one with mkfifo /tmp/live.srt.
- Your script writes SRT lines into it as if it were a file.
- A player can read from it like a file path, effectively “following” new subtitles as they’re
  written.

Players like mpv can reload or continually read subtitles from a FIFO, providing near-live overlay
without writing to disk repeatedly.

### Performance

#### Minimize resampling cost

If the device can be opened at 16 kHz, do so and skip resample.

### Speed & accuracy optimizations

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

### Modularization

When the main script exceeds ~300 lines and mixes concerns. A common split:

- audio_io.py: device discovery, capture, resampling.
- vad_segmenter.py: frame logic, state machine.
- transcriber.py: model init and decode helpers.
- cli.py: argument parsing and presets.
- outputs/srt_writer.py: writers/formatters.
- ui/overlay.py: on-screen overlay code.
- main.py: glues components.
