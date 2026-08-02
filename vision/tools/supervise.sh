#!/bin/sh
# Keep the pipeline alive unattended, and notice when it is alive but not working.
#
# There is no systemd unit because installing one needs root, so this is a plain supervisor
# started at boot from cron (@reboot), which needs no privileges.
#
# It watches for two different failures, and the second is the one that hurt:
#
#   * the process is gone         -- start it;
#   * the process is there and stuck -- restart it. A pipeline blocked on a camera socket
#     that stopped sending never exits and never logs. `pgrep` answers "yes" for as long as
#     you like: there is a 39-hour hole in this log where exactly that happened. So each
#     camera stamps a file with the second it last handled a frame, and a stamp that stops
#     advancing is a wedged pipeline however healthy the process table looks.
#
# Copy to $HOME/supervise.sh; the cron line is:
#     @reboot sleep 30 && /home/markp/supervise.sh >> /home/markp/supervisor.log 2>&1

VISION="$HOME/ccs-v2/vision"
PY="$HOME/coworkingroom-camera-system/venv/bin/python3"
LOG="$HOME/pipeline.log"
SUPLOG="$HOME/supervisor.log"
MAX_LOG_BYTES=20971520  # 20 MB

HEARTBEAT_DIR="$VISION/data/heartbeat"
# A camera delivers 5-15 frames a second and stamps at most once a second. Two minutes of
# silence is far beyond any hesitation and well short of a working day.
STALE_AFTER=120
# Loading the models takes most of a minute on this CPU, and no frame is handled until they
# are up. Without this the staleness check would kill every start before it finished one.
GRACE=240
# Restarting a pipeline whose camera is genuinely unplugged achieves nothing, so back off
# rather than churn. Doubles per consecutive restart, up to this.
BACKOFF_MAX=600

. "$HOME/.profile" >/dev/null 2>&1
cd "$VISION" || exit 1

note() {
    echo "$(date -u '+%Y-%m-%d %H:%M:%SZ') $*" >> "$SUPLOG"
}

running() {
    pgrep -f "stuhi_vision[ ]run" >/dev/null 2>&1
}

# Which camera, if any, has stopped stamping. Prints its name; silence means all is well.
#
# Deliberately fails *open*: if the directory does not exist -- an older build that does not
# stamp at all -- this says nothing rather than declaring everything stale. Failing closed
# there would restart a perfectly healthy pipeline every two minutes for ever, which is a
# worse outcome than not having the check.
wedged() {
    [ -d "$HEARTBEAT_DIR" ] || return 0
    now=$(date +%s)
    for stamp in "$HEARTBEAT_DIR"/*; do
        [ -f "$stamp" ] || continue
        last=$(cat "$stamp" 2>/dev/null || echo 0)
        case "$last" in
            ''|*[!0-9]*) last=0 ;;   # half-written or garbage: treat as never
        esac
        if [ $((now - last)) -gt "$STALE_AFTER" ]; then
            echo "$(basename "$stamp") ($((now - last))s)"
            return 0
        fi
    done
}

start() {
    note "$1 - starting the pipeline"
    nohup env PYTHONPATH="$VISION/src" \
        PYTHONUNBUFFERED=1 \
        TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
        TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
        "$PY" -u -m stuhi_vision run >> "$LOG" 2>&1 &
    sleep 25
    if running; then
        note "pipeline up (pid $(pgrep -f "stuhi_vision[ ]run" | head -1))"
    else
        note "pipeline failed to start - see $LOG"
    fi
    started_at=$(date +%s)
}

rotate() {
    # Rotate rather than truncate in place: the running process holds the file open, so
    # emptying it would leave a sparse file that still counts against the disk.
    if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt "$MAX_LOG_BYTES" ]; then
        note "log exceeded $MAX_LOG_BYTES bytes - keeping the tail"
        tail -c 2000000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
    fi
}

note "supervisor started (pid $$)"
started_at=0
failures=0

while true; do
    if ! running; then
        start "pipeline not running"
        failures=$((failures + 1))
    elif [ $(($(date +%s) - started_at)) -gt "$GRACE" ]; then
        silent=$(wedged)
        if [ -n "$silent" ]; then
            note "no frames from $silent - the pipeline is stuck, restarting it"
            pkill -f "stuhi_vision[ ]run"
            sleep 5
            start "stuck pipeline killed"
            failures=$((failures + 1))
        else
            # A spell of genuine health earns back the short interval.
            if [ "$failures" -gt 0 ]; then
                note "pipeline healthy again after $failures restart(s)"
            fi
            failures=0
        fi
    fi

    rotate

    # Back off while restarts keep being needed, so an unplugged camera costs one line a
    # few minutes rather than one every thirty seconds.
    wait_for=30
    attempt=1
    while [ "$attempt" -lt "$failures" ] && [ "$wait_for" -lt "$BACKOFF_MAX" ]; do
        wait_for=$((wait_for * 2))
        attempt=$((attempt + 1))
    done
    [ "$wait_for" -gt "$BACKOFF_MAX" ] && wait_for="$BACKOFF_MAX"
    sleep "$wait_for"
done
