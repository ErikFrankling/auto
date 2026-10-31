#!/usr/bin/env bash
# Records every video in the report and writes videos/index.json.
#   nix develop -c scripts/record_all.sh [id ...]
# Videos go to videos/ (not committed), with a small web copy in videos/web/.
set -u
A="./target/release/auto --adapter 7900 record"
mkdir -p videos/web
ONLY=" $* "

want() { [ "$ONLY" = "  " ] || [[ "$ONLY" == *" $1 "* ]]; }

# id, name, caption, then the record flags.
rec() {
  id=$1; name=$2; caption=$3; shift 3
  if want "$id"; then
    $A "$id" --title "$name" --out "videos/$id.mp4" "$@" 2>&1 | tail -1 | tee "videos/$id.log"
    ffmpeg -v error -y -i "videos/$id.mp4" -vf scale=-2:720 -c:v libx264 -preset veryfast -crf 28 \
      -maxrate 2500k -bufsize 5000k -pix_fmt yuv420p -threads 3 -movflags +faststart "videos/web/$id.mp4"
    ffmpeg -v error -y -sseof -1 -i "videos/$id.mp4" -frames:v 1 -vf scale=-2:360 -update 1 "videos/web/$id.jpg"
    # Give the desktop a breather between videos.
    sleep 5
  fi
  printf '%s\t%s\t%s\n' "$id" "$name" "$caption" >> videos/.index.tsv
}

rm -f videos/.index.tsv

rec life "Conway's Game of Life" \
  "2.4 billion cells. Starts at 2 pixels per cell, zooms out to the whole world, running one generation per frame." \
  --world 65536x36864 --zoom-start 0.5 --zoom-end 34.2 --seconds 24 --fps 60 --steps-per-frame 1 --warmup 64
rec rule30 "Rule 30" \
  "One live cell in a row of 1920. Each new row is the next generation, so time runs up the screen." \
  --world 1920x1080 --seconds 20 --fps 60 --steps-per-frame 2
rec rule110 "Rule 110" \
  "The 1D rule proved Turing complete. Gliders collide over a repeating background." \
  --world 1920x1080 --seconds 20 --fps 60 --steps-per-frame 2
rec brain "Brian's Brain" \
  "Three states stored in two bit planes. Almost everything is in motion, and ships keep flying out." \
  --world 1920x1080 --seconds 20 --fps 30 --steps-per-frame 1 --warmup 200
rec wireworld "Wireworld" \
  "A generated circuit board with ring clocks feeding random wiring. Blue and white are electrons on copper." \
  --world 960x544 --width 1920 --height 1088 --zoom-start 0.5 --zoom-end 0.5 --seconds 20 --fps 30 --steps-per-frame 1 --warmup 200
rec cyclic "Cyclic cellular automaton" \
  "16 colours in a ring. From noise, spiral 'demons' form and take over the whole world." \
  --world 960x540 --zoom-start 0.5 --zoom-end 0.5 --seconds 20 --fps 30 --steps-per-frame 1
rec hodgepodge "Hodgepodge machine" \
  "A model of the Belousov-Zhabotinsky reaction. Waves of infection rise, burn out and start again." \
  --world 960x540 --zoom-start 0.5 --zoom-end 0.5 --seconds 20 --fps 30 --steps-per-frame 1 --warmup 300
rec ants "Langton's ants" \
  "LLRR turmites, about 250 of them. As a true CA: each cell holds a 4-bit mask of the ants on it, so they never collide." \
  --world 1920x1080 --seconds 20 --fps 60 --steps-per-frame 150
rec forest "Forest fire" \
  "Trees grow, lightning strikes, fire spreads. A stochastic rule, with per-cell random numbers from a hash." \
  --world 1920x1080 --seconds 20 --fps 30 --steps-per-frame 1 --warmup 1500
rec sandpile "Abelian sandpile" \
  "A million grains on one cell. Any cell holding 4 gives one to each neighbour, and a fractal grows out." \
  --world 1024x1024 --width 1080 --height 1080 --seconds 20 --fps 60 --steps-per-frame 4000
rec sand "Falling sand" \
  "A Margolus block automaton: in-place, one buffer, 2x2 blocks on a grid that shifts every tick. Sand and water through rock." \
  --world 1920x1080 --seconds 24 --fps 60 --steps-per-frame 8
rec grayscott "Gray-Scott reaction-diffusion" \
  "Feed and kill rates change across the world, so spots, worms and mazes all appear in one picture." \
  --world 1920x1080 --seconds 20 --fps 60 --steps-per-frame 24 --warmup 500
rec lenia "Lenia" \
  "Orbium, the best known Lenia creature, from Bert Chan's catalogue. Each step is a pair of 2D FFTs, and the creatures glide and collide." \
  --world 512x256 --width 1920 --height 960 --zoom-start 0.2667 --zoom-end 0.2667 --seconds 20 --fps 60 --steps-per-frame 1
rec smoothlife "SmoothLife" \
  "Rafler's continuous Game of Life. Discs of life spread into a shifting labyrinth." \
  --world 1024x512 --width 1920 --height 1024 --zoom-start 0.5 --zoom-end 0.5 --seconds 20 --fps 30 --steps-per-frame 1
rec mnca "Multiple Neighbourhood CA" \
  "Threshold bands on two neighbourhood averages. Worm-like blobs push each other around." \
  --world 1024x512 --width 1920 --height 1024 --zoom-start 0.5 --zoom-end 0.5 --seconds 20 --fps 30 --steps-per-frame 1
rec physarum "Physarum slime mould" \
  "1 million agents follow and lay trails over a diffusing field. Particles writing to a grid, with atomics." \
  --world 1920x1080 --seconds 24 --fps 60 --steps-per-frame 2

python3 - <<'PY'
import json
rows = []
for line in open('videos/.index.tsv'):
    i, n, c = line.rstrip('\n').split('\t')
    rows.append({'id': i, 'name': n, 'caption': c, 'web': f'videos/{i}.mp4', 'poster': f'videos/{i}.jpg'})
json.dump(rows, open('videos/index.json', 'w'), indent=1)
PY
