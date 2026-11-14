"""Engine demo of the AI's own view for the frag movie (scripts/fragmovie.py renders it).

The AI's engine records an ordinary Xash demo (`record` / `stop` on its own client): every update its client
receives, with the view angles the bridge set, the entities and the HUD messages. Recording only writes what the
client already receives; the frames the model sees are not touched.

One demo per duel.py process (one end of a session), started right after the AI's server is up and BEFORE anyone
joins. (Starting a recording mid-game while a human client is connected gives demos the engine cannot play back:
the second update already fails with "CL_ParsePacketEntities: overflow". A recording that is running when the
human joins plays back fine.)

Session directory layout (both ends of a session share one directory):
    <dir>/T_195512.dem     the AI's view for one end (side, start time of that end)
    <dir>/rounds.jsonl     one line per finished round: result, kill time on the demo clock, running score
If the process dies before it can move the demo, the demo stays in the engine dir under rounds.jsonl's
"engine_tmp" name and the renderer picks it up from there.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path


class DemoRecorder:
    def __init__(self, bridge_fn, session_dir: str, model_side: str, opponent: str, game_dir: Path):
        self.bridge_fn = bridge_fn          # () -> EngineBridge (the env boots its engine lazily)
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.side = model_side
        self.opponent = opponent
        self.cstrike = Path(game_dir) / "cstrike"
        self.name = f"{model_side}_{time.strftime('%H%M%S')}.dem"
        self.tmp = f"fm{os.getpid() % 100000}"   # engine-side name while recording (short, plain)
        self.recording = False
        self.t_rec = None

    def begin(self):
        b = self.bridge_fn()
        b.command(f"record {self.tmp}")
        self.recording = True
        self.t_rec = float(b.state.time)

    def log_round(self, rnd: int, result: str, t_start: float, t_end: float, score: dict, round_time: float):
        row = dict(demo=self.name, engine_tmp=f"{self.tmp}.dem", round=rnd, model_side=self.side,
                   opponent=self.opponent, result=result, t_rec=round(self.t_rec, 4), t_start=round(t_start, 4),
                   t_end=round(t_end, 4), round_time=round(round_time, 3), ai=score["ai"], opp=score["you"],
                   draws=score["draw"], wall=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.dir / "rounds.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def end(self):
        """Stop recording (if the engine still answers)."""
        if not self.recording:
            return
        self.recording = False
        try:
            self.bridge_fn().command("stop", steps=2)
        except Exception:
            pass                            # engine gone: it closes the demo when it quits

    def keep(self):
        """After the engine has quit: move the finished demo into the session directory."""
        src = self.cstrike / f"{self.tmp}.dem"
        if src.exists():
            shutil.move(str(src), str(self.dir / self.name))
        return self.dir / self.name
