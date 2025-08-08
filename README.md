# Live Captions

You can overlay subtitle with mpv:

    ./live-captions.py --source system --model small.en > /tmp/live.srt &
    mpv --sub-file=/tmp/live.srt /path/to/Battlestar_Galactica.mkv

mpv auto-reloads the SRT every few seconds; captions appear on-screen.
