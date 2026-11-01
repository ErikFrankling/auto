#!/usr/bin/env bash
# Builds the life report with its media and publishes it to the site's life/ folder.
#   nix develop -c scripts/build_life_report.sh [--publish]
# Media are phone-sized copies of what's in videos/life-gpu. Only missing ones are made.
set -eu
cd "$(dirname "$0")/.."
V=videos/life-gpu
M=report/life/media
mkdir -p "$M"

clip() { # source name: small enough to play on a phone over a mobile connection
  if [ -f "$1" ] && { [ ! -f "$M/$2.mp4" ] || [ "$1" -nt "$M/$2.mp4" ]; }; then
    nice -n 19 ffmpeg -v error -y -i "$1" -vf scale=1280:720 -c:v libx264 -preset veryfast -crf 30 -maxrate 4M -bufsize 8M \
      -pix_fmt yuv420p -movflags +faststart -threads 2 "$M/$2.mp4"
  fi
}
still() { # source name filter
  if [ -f "$1" ] && { [ ! -f "$M/$2.jpg" ] || [ "$1" -nt "$M/$2.jpg" ]; }; then
    ffmpeg -v error -y -i "$1" -vf "$3" -q:v 3 "$M/$2.jpg"
  fi
}
clip $V/giant/4096.mp4 world4096
clip $V/giant/2048.mp4 world2048
clip $V/giant/8192.mp4 world8192
clip $V/closeup/seed1.mp4 close-seed1
clip $V/closeup/seed3.mp4 close-seed3
clip $V/closeup/world4096.mp4 close-world4096
[ -f $V/giant/4096-zoom.jpg ] && cp -u $V/giant/4096-zoom.jpg "$M/zoom4096.jpg"
still $V/giant/2048.png world2048 "scale=1024:1024:flags=area"
still $V/giant/4096.png world4096 "scale=1400:1400:flags=area"
for n in 1 2 3; do
  for prefix in v2 old; do
    if [ -f "$V/maps/$prefix-seed$n-lifestyles.png" ]; then
      still "$V/maps/$prefix-seed$n-lifestyles.png" "map-seed$n" "scale=1024:1024:flags=neighbor"
      break
    fi
  done
done
for n in 2048 4096; do
  [ -f "$V/giant/$n.png" ] && cp -u "$V/giant/$n.png" "$M/world$n-full.png"
done
python3 scripts/make_life_report.py
du -sh report/life

if [ "${1:-}" = "--publish" ]; then
  W=$(mktemp -d /tmp/auto-pages.XXXXXX)
  git fetch -q origin gh-pages
  git worktree add -q --detach "$W" FETCH_HEAD
  mkdir -p "$W/life"
  cp report/life/index.html "$W/life/index.html"
  cp -r "$M" "$W/life/"
  (
    cd "$W"
    git add -A life
    if git diff --cached --quiet; then
      echo "nothing new to publish"
    else
      git commit -qm "Update life report"
      git -c 'credential.helper=!gh auth git-credential' push -q origin HEAD:gh-pages
      echo "published"
    fi
  )
  git worktree remove --force "$W"
fi
