#!/usr/bin/env bash
# The one process that uses the GPU for life runs. Takes the lock once, runs jobs in
# order for up to a turn, then gives the other projects a go.
spool=/tmp/life-spool
TURN=${LIFE_TURN_SECONDS:-720}
JOB_MINUTES=${LIFE_JOB_MINUTES:-5}
exec 8>/tmp/auto-gpu.lock
exec 9>/tmp/cars-gpu.lock
pending() { ls "$spool/jobs" 2>/dev/null | grep -v '\.pid$' | sort | head -1; }
while true; do
  if [ -z "$(pending)" ]; then sleep 3; continue; fi
  flock 8
  flock 9
  began=$(date +%s)
  echo "$(date +%T) got the GPU"
  while true; do
    job=$(pending)
    [ -z "$job" ] && break
    # Only start what can finish inside this turn.
    [ $(( $(date +%s) - began + JOB_MINUTES * 60 )) -gt "$TURN" ] && break
    pid=$(cat "$spool/jobs/$job.pid" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$spool/jobs/$job" "$spool/jobs/$job.pid"
      echo 130 > "$spool/done/$job"
      continue
    fi
    echo "$(date +%T) job $job: $(tail -1 "$spool/jobs/$job" | cut -c1-200)"
    LIFE_LOCK_HELD=1 LIFE_MAX_MINUTES=$JOB_MINUTES bash "$spool/jobs/$job" >> "$spool/logs/$job.log" 2>&1
    code=$?
    rm -f "$spool/jobs/$job" "$spool/jobs/$job.pid"
    echo "$code" > "$spool/done/$job"
    echo "$(date +%T) job $job finished with $code"
  done
  flock -u 9
  flock -u 8
  echo "$(date +%T) gave the GPU back"
  sleep 1
done
