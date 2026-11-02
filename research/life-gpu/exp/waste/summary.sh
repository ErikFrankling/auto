#!/usr/bin/env bash
# One line per log entry, with lifestyle numbers weighted by each group's share of cells.
for f in "$@"; do
  jq -r --arg f "$(basename "$f" .jsonl)" 'select(.tick) as $r | ($r.lifestyles | map(.share) | add) as $t
    | def w(k): ($r.lifestyles | map(.share * k) | add) / $t;
    [$f, $r.tick, $r.cells, ($r.density*1000|round/1000), ($r.effective_species*10|round/10),
     ($r.species_at["0.3"]*10|round/10), ($r.species_at["0.5"]*10|round/10),
     $r.species_explain_behaviour, $r.groups_1pct, ($r.lifestyles|length),
     (w(.moving)*1000|round/1000), (w(.spread)*1000|round/1000), (w(.neighbours)*100|round/100), (w(.heads_to_food)*1000|round/1000),
     (w(.budget.x0)*1000|round/1000), (($r.deaths.bitten / ($r.deaths|add))*100|round), (($r.deaths.burned / ($r.deaths|add))*100|round)] | @tsv' "$f"
done | column -t -N file,tick,cells,dens,es,sp.3,sp.5,seb,g1pct,life,moving,spread,neigh,tofood,x0,bit%,burn%
