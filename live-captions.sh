#!/usr/bin/env bash

# live-captions.sh - Set up prerequisites and run the live captions application

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install required packages
pip install -q -r requirements.txt

# Warn if blackhole-2ch is not installed
if ! system_profiler SPAudioDataType | grep -q "BlackHole"; then
  echo "[WARN]: BlackHole audio driver is not installed."
  echo "        Install with: \`brew install blackhole-2ch\` and restart."
fi

# Run the live captions application
./live-captions.py "$@"
