#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 start_ui.py
else
  echo 'Please install Python 3.10–3.13 (recommended: 3.12) and try again.'
fi
read -r -p 'Press Enter to close this terminal. '
