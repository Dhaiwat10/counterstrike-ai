#!/bin/bash
# usage: smoke.sh NAME extra-flags...   (scratch run, 2 envs, 6 min, from the pre-lrhalf checkpoint at 120 ms)
name=$1; shift
cd "$(dirname "$0")/.."
timeout 900 .venv/bin/python -m train.ppo --run $name --resume checkpoints/aim_640/pre_lrhalf_last.pt --lr 1e-4 --hours 0.1 \
  --ent-yaw-mult 5.0 --rehearsal 0.1 --envs 2 --rollout 32 --resume-warmup 0 --offtarget-cost 0.0015 --acquire-bonus 0.3 \
  --rand-delay 0.25 --no-contest 0.5 --leash 150 --leash-frac 0.8 --min-vis 0.75 --all-rounds 11 \
  --delay-ramp 0.04,0.08,0.1,0.12,0.16,0.2 --stage0 15 "$@" 2>&1 | grep -E '"event": "(update|done)"' | python3 -c "
import sys,json
rows=[json.loads(l) for l in sys.stdin if 'update' in l]
for e in rows[::6]+rows[-1:]:
    print('$name', {k:(round(e[k],3) if isinstance(e.get(k),float) else e.get(k)) for k in ['update','kill_rate','episodes','teacher','teacher_agree','aux','ent','fire_frac'] if k in e})
"
