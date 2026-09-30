#!/usr/bin/env bash
# Generates a dance's clips a few a day, on Hugging Face's free ZeroGPU quota, until the chain is complete.
#
# ZeroGPU gives about three LTX clips a day, so a seven-section dance takes days. This runs
# `dance.py <spec> clips --ltx --chain <chain>` every day at the given time (after the quota resets); each
# run carries on from the first missing clip and stops cleanly when the quota runs out. When every clip of
# the chain is there, the job uninstalls itself. It makes clips only: the edit and its frame-by-frame
# check are left for a person.
#
#   ./install_dance_schedule.sh dances/mon-happy.json B 19:45    install and start
#   ./install_dance_schedule.sh dances/mon-happy.json remove     uninstall
#
# launchd rather than cron for the same reason as install_schedule.sh: it catches up a run the Mac slept through.

set -euo pipefail

SPEC="${1:?usage: $0 dances/<name>.json <chain>|remove [HH:MM]}"
NAME="$(basename "$SPEC" .json)"
LABEL="com.pawfect.dance.$NAME"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT/.venv/bin/python"

if [[ "${2:-}" == "remove" ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL"
  exit 0
fi

CHAIN="${2:?usage: $0 dances/<name>.json <chain>|remove [HH:MM]}"
AT="${3:-19:45}"
HOUR=$((10#${AT%%:*}))
MINUTE=$((10#${AT##*:}))
LOG="$PROJECT/logs/dance-$NAME.log"

if [[ ! -x "$PYTHON" ]]; then
  echo "No virtualenv at $PYTHON"
  exit 1
fi
mkdir -p "$PROJECT/logs" "$HOME/Library/LaunchAgents"

# One run: make what the quota allows, then uninstall if the chain is complete.
read -r -d '' JOB <<JOBEOF || true
cd "$PROJECT"
echo "== \$(date)"
"$PYTHON" dance.py "$SPEC" clips --ltx --chain "$CHAIN"
if "$PYTHON" -c "import sys; sys.argv=['dance.py']; import dance; n, s = dance.load_spec('$SPEC'); from pipeline.config import slot_dir; ch = next(c for c in s['chains'] if c['name'] == '$CHAIN'); sys.exit(len(dance.chain_clips(s, slot_dir(n), '$CHAIN')) < len(dance.links(s, ch)))"; then
  echo "chain $CHAIN complete - next: $PYTHON dance.py $SPEC edit --chain $CHAIN"
  "$PROJECT/install_dance_schedule.sh" "$SPEC" remove
fi
JOBEOF

cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-c</string>
    <string><![CDATA[$JOB]]></string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
  <key>RunAtLoad</key><false/>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
PLISTEOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Installed $LABEL"
echo "  runs:  every day at $(printf '%02d:%02d' "$HOUR" "$MINUTE") until chain $CHAIN of $SPEC is complete"
echo "  logs:  $LOG"
