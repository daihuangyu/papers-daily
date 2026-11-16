#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 scripts/update_index.py
git add -- README.md sources.json VLM scripts AUTOMATION.md requirements.txt .gitignore
if git diff --cached --quiet; then
  echo 'No new content to publish.'
  exit 0
fi
git commit -m "docs: update daily illustrated research notes"
GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15' git push origin main
