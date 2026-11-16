#!/bin/sh
# Optional for a managed cloud checkout with an already-authorized HTTPS origin.
# The scheduled workflow normally publishes through the GitHub connector; see AUTOMATION.md.
set -eu
cd "$(dirname "$0")/.."
python3 scripts/update_index.py
if [ "$(git branch --show-current)" != main ]; then
  echo 'Refusing to publish a branch other than main.' >&2
  exit 1
fi
case "$(git remote get-url origin)" in
  https://github.com/daihuangyu/papers-daily.git|https://github.com/daihuangyu/papers-daily) ;;
  *) echo 'Cloud publication requires the authorized HTTPS repository; use the GitHub connector instead of SSH.' >&2; exit 1 ;;
esac
git add -- README.md sources.json VLM scripts AUTOMATION.md requirements.txt
if ! git diff --cached --quiet; then
  git commit -m "docs: update daily illustrated research notes"
fi
# No credential prompts, force push, SSH override or copied local credentials.
GIT_TERMINAL_PROMPT=0 git push origin main
