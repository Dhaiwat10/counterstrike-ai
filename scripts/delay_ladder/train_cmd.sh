#!/bin/bash
# usage: train_cmd.sh RUN RESUME ENVS [extra args...]   -- the peek phase (stay home / open is deadly), both ends.
# Base = scratch/sides/train_cmd3.sh (sides_v6) with, for this phase:
#   removed  --route-shaping 0.0003 --still-cost 0.001 (they push the model forward), --camp-diff0, --duel-max-len 16 (camp only)
#   added    --duel-camp-frac 0 (the stays-put shooter is retired), --duel-stage0 2 (peeker difficulty 0 first),
#            --time-penalty-round 1.0 (a time-out costs as much as a death), --open-switch (+ --open-switch-frac),
#            --nav-goal home, --duel-hold-frac, --arena-hold-frac; kept --skin-adapt 4 --arena-wall-frac 0.4 (sides_v6)
cd "$(dirname "$0")/../.."  # the project root (was a hard-coded Mini path)
P=$PWD/.venv/bin/python
export PYTHONPATH=$PWD
RUN=$1; RES=$2; ENVS=$3; shift 3
exec $P -m train.ppo --run $RUN --scenario duel_aimmap,aim_aimmap,duel_aimmap,aim_aimmap,duel_aimmap \
  --resume $RES --resume-warmup 2 --no-delay --stage0 12 --envs $ENVS --lr 3e-5 --kl-prior checkpoints/bc_fov80/best.pt --kl-coef 0.01 \
  --rand-delay 0 --rehearsal 0 --no-contest 0.5 --leash 150 --leash-frac 0.8 --min-vis 0.75 --offtarget-cost 0.0015 --acquire-bonus 0.3 \
  --teacher-coef 0.02 --teacher-min 0.02 --teacher-soft --teacher-aim-only --teacher-near-mult 50 --teacher-near-deg 2 --max-dist 1800 \
  --passive-frac 0.1 --teacher-pitch-deadband 0.4 --snapshot-every 25 --heads yaw,pitch,attack,fwd,side --nav-coef 0.6 --nav-coef-min 0.4 \
  --nav-decay-steps 2e6 --nav-look-deadband 2 --nav-look-follow 100 --selfloc 1.0 --startend 0.5 --duel-random-spawn \
  --duel-gates 0.6,0.65,0.6 --duel-spawn-adapt 4 --search-flip-cost 0.005 --sides T,CT --skin-change 0.25 --cur-down 0 \
  --skin-adapt 4 --arena-wall-frac 0.4 --duel-camp-frac 0 --duel-stage0 2 --time-penalty-round 1.0 \
  --open-switch --nav-goal home --duel-hold-frac 0.5 --arena-hold-frac 0.3 "$@"
