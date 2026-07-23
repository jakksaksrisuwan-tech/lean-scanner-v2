#!/bin/zsh
# Start (or restart) the lean-scanner quick tunnel on port 8081.
# Fully isolated: never touches other cloudflared processes (the
# token-run tunnel or quick tunnels on other ports).
# Usage: ./scripts/tunnel.sh    → prints the fresh URL when ready.
set -e
PORT=8081
LOG=/tmp/lean-scanner-tunnel.log

# ensure the app server is up
if ! curl -s -o /dev/null "http://localhost:$PORT/"; then
  echo "server not running — starting python3 server.py $PORT"
  (cd "$(dirname "$0")/.." && nohup python3 server.py $PORT > /tmp/lean-scanner-server.log 2>&1 &)
  sleep 2
fi

# kill only OUR previous quick tunnel (exact port match), nothing else
pkill -f "cloudflared tunnel --url http://localhost:$PORT" 2>/dev/null || true
sleep 1

nohup cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate > "$LOG" 2>&1 &
echo "waiting for tunnel..."
for i in {1..30}; do
  URL=$(grep -o "https://[a-z-]*\.trycloudflare\.com" "$LOG" | head -1)
  [ -n "$URL" ] && break
  sleep 2
done
if [ -z "$URL" ]; then
  echo "tunnel failed to start — see $LOG"
  exit 1
fi
echo ""
echo "  app:  $URL/"
echo "  dump: $URL/src/camera_dump.html"
echo ""
echo "note: quick-tunnel URLs rotate on every restart. For a permanent"
echo "URL, run 'cloudflared tunnel login' once (interactive) and ask"
echo "Claude to set up a named tunnel on your domain."
