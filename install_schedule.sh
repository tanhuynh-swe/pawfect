#!/usr/bin/env bash
# Installs the unattended schedule on macOS using launchd.
#
# launchd, not cron: cron jobs on macOS die quietly when the Mac sleeps, and
# launchd will catch up a missed run when the machine wakes. For a Mac mini
# that is exactly what you want.
#
#   ./install_schedule.sh          install and start
#   ./install_schedule.sh remove   uninstall
#
# Runs `run.py auto` at 08:00 on Tuesday, Thursday and Saturday.
# Change the hours/weekdays below if you want a different cadence.

set -euo pipefail

LABEL="com.pawfect.autopublish"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT/.venv/bin/python"

if [[ "${1:-}" == "remove" ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL"
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "No virtualenv at $PYTHON"
  echo "Run:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

mkdir -p "$PROJECT/logs" "$HOME/Library/LaunchAgents"

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
    <string>$PYTHON</string>
    <string>$PROJECT/run.py</string>
    <string>auto</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <!-- Without this, Python block-buffers its output when not attached to a
         terminal, so the log stays empty until the run ends and you cannot
         watch progress or see where a failure happened. -->
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>StandardOutPath</key><string>$PROJECT/logs/auto.log</string>
  <key>StandardErrorPath</key><string>$PROJECT/logs/auto.log</string>
  <key>RunAtLoad</key><false/>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
PLISTEOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Installed $LABEL"
echo "  runs:  Tue / Thu / Sat at 08:00"
echo "  logs:  $PROJECT/logs/auto.log"
echo ""
echo "Test it right now without waiting:"
echo "  launchctl kickstart -p gui/$(id -u)/$LABEL"
echo ""
echo "IMPORTANT: stop the Mac mini from sleeping through its schedule:"
echo "  sudo pmset -a sleep 0 disksleep 0"
