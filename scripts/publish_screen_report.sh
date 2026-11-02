#!/usr/bin/env bash
# Publishes the screen results page to the site's life/ folder.
#   scripts/publish_screen_report.sh
set -eu
cd "$(dirname "$0")/.."
W=$(mktemp -d /tmp/auto-pages.XXXXXX)
git fetch -q origin gh-pages
git worktree add -q --detach "$W" FETCH_HEAD
mkdir -p "$W/life/media"
cp report/life/screen.html "$W/life/screen.html"
rm -rf "$W/life/media/screen"
cp -r report/life/media/screen "$W/life/media/"
(
  cd "$W"
  git add -A life
  if git diff --cached --quiet; then
    echo "nothing new to publish"
  else
    git commit -qm "Update screen results"
    git -c 'credential.helper=!gh auth git-credential' push -q origin HEAD:gh-pages
    echo "published"
  fi
)
git worktree remove --force "$W"
