#!/usr/bin/env bash
# Assembles site/: the report, its videos and the browser app.
#   nix develop -c scripts/build_site.sh
set -eu
python3 scripts/make_report.py
(cd crates/auto-web && trunk build --release --public-url ./ >/dev/null 2>&1)
rm -rf site
mkdir -p site/app site/videos
cp report/index.html site/index.html
cp crates/auto-web/dist/* site/app/
cp videos/web/* site/videos/
touch site/.nojekyll
du -sh site
