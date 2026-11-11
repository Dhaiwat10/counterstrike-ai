"""Gymnasium environment for Counter-Strike 1.6 with human-equivalent constraints.

Observation: the rendered screen (RGB uint8, HxWx3) exactly as a player would see it
(world + HUD), delivered `reaction_delay` seconds late.

Action (MultiDiscrete), applied for `action_repeat` engine ticks:
  0 yaw    : turn bins (degrees per agent step, spread evenly over the repeated ticks)
  1 pitch  : look up/down bins
  2 fwd    : 0 none, 1 forward, 2 back
  3 side   : 0 none, 1 left, 2 right
  4 attack : 0/1
  5 jump   : 0/1
  6 duck   : 0/1
  7 walk   : 0/1 (shift: quiet + accurate)
  8 reload : 0/1

Fairness layer (defaults chosen from DeepMind FTW / OpenAI Five precedent and our pro
POV demo stats, see PHASE0_RESULTS.md):
  * reaction_delay = 0.20 s observation latency
  * max turn rate  = 1200 deg/s (dsn's 99.9th percentile per-frame yaw speed ~1160 deg/s)
  * decision rate  = 1 / (tick * action_repeat) = 25 Hz by default
"""
from __future__ import annotations

import collections
import math
import zlib
from pathlib import Path

import gymnasium as gym
import numpy as np

from .bridge import (EngineBridge, EngineConfig, IN_ATTACK, IN_BACK, IN_DUCK, IN_FORWARD,
                     IN_JUMP, IN_MOVELEFT, IN_MOVERIGHT, IN_RELOAD, IN_RUN)
from .scenarios import SCENARIOS, Scenario

YAW_BINS = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, 24, 36, 48], dtype=np.float32)
PITCH_BINS = np.array([0, 0.25, 1, 3, 8, 16], dtype=np.float32)
YAW_TABLE = np.concatenate([-YAW_BINS[:0:-1], YAW_BINS])      # 21 symmetric bins
PITCH_TABLE = np.concatenate([-PITCH_BINS[:0:-1], PITCH_BINS])  # 11 symmetric bins
WALK_SCALE = 0.52
RUN_SPEED = 250.0  # CS maxspeed with rifles is <= 250; engine clamps to weapon speed


def pockets_path(arena: str):
    """data/arena/dust2_pairs.npz -> data/arena/dust2_pockets.npz (None if the name has no '_pairs')."""
    p = Path(arena)
    return p.with_name(p.name.replace("_pairs", "_pockets")) if "_pairs" in p.name else None


def arena_crc(positions, pairs) -> int:
    """Fingerprint of an arena's spawn data, stored in the pockets file so a stale one is never used."""
    return zlib.crc32(np.ascontiguousarray(pairs).tobytes(), zlib.crc32(np.ascontiguousarray(positions).tobytes()))


class CS16Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenario: str | Scenario = "dm_dust2",
        width: int = 320,
        height: int = 240,
        renderer: str = "gl",
        tick: float = 0.01,
        action_repeat: int = 4,
        reaction_delay: float = 0.20,
        max_turn_rate: float = 1200.0,
        kill_reward: float = 1.0,
        death_penalty: float = 1.0,
        damage_reward: float = 0.01,
        hidden: bool = True,
        engine_log: bool = False,
    ):
        super().__init__()
        self.scn = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.cfg = EngineConfig(width=width, height=height, renderer=renderer, tick=tick,
                                hidden=hidden, maxplayers=max(2, self.scn.bots + 2), log=engine_log)
        self.tick = tick
        self.repeat = action_repeat
        self.delay_steps = int(round(reaction_delay / (tick * action_repeat)))
        self.max_turn_per_tick = max_turn_rate * tick
        self.rw_kill, self.rw_death, self.rw_dmg = kill_reward, death_penalty, damage_reward

        self.observation_space = gym.spaces.Box(0, 255, (height, width, 3), np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([len(YAW_TABLE), len(PITCH_TABLE), 3, 3, 2, 2, 2, 2, 2])

        self.bridge: EngineBridge | None = None
        self._arena = None
        self._pockets = None  # precomputed pocket visibility per (pair, orientation); see _pocket()
        if self.scn.arena:
            z = np.load(self.scn.arena)
            self._arena = (z["positions"], z["pairs"], z["dist"])
            pk = pockets_path(self.scn.arena)
            if pk is not None and pk.exists():
                zp = np.load(pk)
                if int(zp["crc"]) == arena_crc(z["positions"], z["pairs"]):  # built for exactly these spawn pairs
                    self._pockets = (zp["vis"], zp["mask"])
        self.arena_max_dist = self.scn.arena_max_dist
        # training-only shaping: reward per degree the crosshair moves toward the nearest live enemy
        # (potential-based, Phi = -angular error; computed from server state, never observed by the agent)
        self.aim_shaping = 0.0
        self.bullet_cost = 0.0         # training only: reward per step spent holding fire
        self.time_penalty = 0.0        # training only: reward per agent step until the round ends
        self.hurt_penalty = 0.0        # training only: reward per HP the agent loses
        self.shaping_gamma = 0.99      # potential-based shaping uses gamma * Phi(s') - Phi(s)
        # training only, teach FLICKS: every step costs offtarget_cost per degree of aim error (creeping accumulates far
        # more than snapping), and the first time the crosshair gets within acquire_deg of the enemy in a round pays
        # acquire_bonus * max(0, 1 - t / acquire_window) (a fast snap pays, a slow creep pays ~nothing)
        self.offtarget_cost = 0.0
        self.acquire_bonus = 0.0
        self.acquire_deg = 3.0
        self.acquire_window = 1.0
        self._acquired_t = None
        # arena: end the round as a NO-CONTEST if the bot stays out of line of sight this long (a free-roaming bot
        # walking behind a wall makes the round unwinnable); 0 = off
        self.no_contest_after = 0.0
        self._hidden_steps = 0
        # arena, free-roaming bot (level 3): LEASH -- YaPB walks its own navigation route away from the spot (median 200
        # units in a straight line at ~216 u/s) and turns a corner; the round is then a no-contest (38% of 800-1050 unit
        # rounds). If the bot gets further than arena_leash units from its spot it is put back onto a random visible,
        # walkable point of the pocket around the spot, facing the agent; it keeps aiming, shooting and walking by itself.
        # arena_leash_frac = share of level-3 rounds that are leashed (the rest stay fully free-roaming); 0 = off.
        self.arena_leash = 0.0
        self.arena_leash_frac = 1.0
        # arena: reject spawn pairs whose pocket (24 points, radii 64-192 around the bot's spot) is less than this
        # fraction visible from the agent's eye (reachable points only); 0 = accept every pair
        self.arena_min_vis = 0.0
        self._leash = None
        self._pocket_vis = None
        self.episode_seconds = self.scn.episode_seconds  # can grow with the arena distance
        self._ep_dist = 0.0
        self.bot_level = 3             # arena: 0 frozen & passive, 1 strafing (scripted) & passive, 2 frozen & shoots, 3 normal
        self._strafe = None
        self._bot_level_applied = None
        self.arena_max_offset = self.scn.arena_max_offset
        self.arena_min_offset = 0.0  # curriculum: starting yaw error is drawn from [min, max] (bounded range)
        self.arena_min_dist = 0.0     # distance stages: newest band starts here...
        self.arena_band_frac = 0.0    # ...and this share of rounds is drawn from (min_dist, max_dist]
        self.arena_review_frac = 0.0  # share of "review" rounds drawn from easier angles [0, min] (anti-forgetting)
        self._ep_offset = 0.0
        self._obs_queue: collections.deque = collections.deque()
        self._blank = np.zeros((height, width, 3), np.uint8)

    # ------------------------------------------------------------------ engine lifecycle
    def _boot(self):
        self.bridge = EngineBridge(self.cfg)
        b = self.bridge
        # crosshair: the default "auto" size rounds to nothing at 320x240; "large" solid green is the
        # standard CS crosshair at the same fraction of the screen as on a 1080p monitor (~4-5 px)
        b.command("cl_crosshair_size large; cl_crosshair_translucent 0; cl_crosshair_color \"50 250 50\"; crosshair 1")
        for c in self.scn.setup_commands():
            b.command(c)
        b.command(f"map {self.scn.map}")
        b.wait_until(lambda s: s.in_game and s.playernum >= 0, max_steps=100000)
        # settle: bots connect, everyone spawns
        b.wait_until(lambda s: self._me(s).alive and self._bots_ready(s), max_steps=3000)

    def _me(self, s):
        return s.players[s.playernum] if 0 <= s.playernum < len(s.players) else s.players[0]

    def _others_alive(self, s):
        return all(p.alive for i, p in enumerate(s.players) if p.connected and i != s.playernum)

    def _bots_ready(self, s):
        n = sum(1 for i, p in enumerate(s.players) if p.connected and i != s.playernum)
        return n >= self.scn.bots

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.bridge is None:
            self._boot()
        elif self.scn.reset_round:
            # fresh round: both sides respawn at their spawn points
            t_restart = self.bridge.state.time
            self.bridge.command("sv_restart 1")
            self.bridge.wait_until(lambda s: s.time > t_restart + 1.2 and self._me(s).alive
                                   and self._bots_ready(s) and self._others_alive(s), max_steps=5000)
        b = self.bridge
        s = b.state
        # wait until we're alive (respawn is automatic in all scenarios)
        if not self._me(s).alive:
            s = b.wait_until(lambda s: self._me(s).alive, max_steps=2000)
        if self._arena is not None:
            self._apply_bot_level()
            s = self._arena_place()
        s = b.step(apply=True, capture=True)  # one neutral step to get a fresh frame
        self._ep_start = s.time
        self._acquired_t = None
        self._hidden_steps = 0
        self._snap = self._snapshot(s)
        self._kills = self._deaths = 0
        self._damage = 0
        frame = b.frame()
        first = frame if frame is not None else self._blank
        self._obs_queue = collections.deque([first] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        # privileged labels aligned with the (delayed) observation, for auxiliary training losses only
        lab = self._target(s)
        self._aux_queue = collections.deque([lab] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        return self._obs_queue[0], self._info(s)

    def step(self, action):
        a = np.asarray(action, dtype=np.int64)
        yaw = float(YAW_TABLE[a[0]])
        pitch = float(PITCH_TABLE[a[1]])
        fwd = (0.0, RUN_SPEED, -RUN_SPEED)[a[2]]
        side = (0.0, -RUN_SPEED, RUN_SPEED)[a[3]]
        if a[7]:  # walk = client-side speed scaling, exactly like +speed (cl_movespeedkey 0.52)
            fwd *= WALK_SCALE
            side *= WALK_SCALE
        buttons = 0
        buttons |= (0, IN_FORWARD, IN_BACK)[a[2]]
        buttons |= (0, IN_MOVELEFT, IN_MOVERIGHT)[a[3]]
        buttons |= IN_ATTACK if a[4] else 0
        buttons |= IN_JUMP if a[5] else 0
        buttons |= IN_DUCK if a[6] else 0
        buttons |= IN_RUN if a[7] else 0
        buttons |= IN_RELOAD if a[8] else 0

        # spread the mouse motion over the repeated ticks, capped at the human turn rate
        dyaw = float(np.clip(yaw / self.repeat, -self.max_turn_per_tick, self.max_turn_per_tick))
        dpitch = float(np.clip(-pitch / self.repeat, -self.max_turn_per_tick, self.max_turn_per_tick))

        b = self.bridge
        reward = 0.0
        terminated = False
        aim0 = self._snap["aim"]
        for i in range(self.repeat):
            cmds = [self._strafe_tick(), self._leash_tick(),
                    self._los_cmd() if i == self.repeat - 1 and self.no_contest_after else ""]
            s = b.step(dyaw=dyaw, dpitch=dpitch, forward=fwd, side=side, buttons=buttons,
                       capture=(i == self.repeat - 1), cmd=";".join(c for c in cmds if c))
            r, died, killed = self._account(s)
            reward += r
            if (died and self.scn.end_on_death) or (killed and self.scn.end_on_kill):
                terminated = True
        if self.aim_shaping:  # potential-based: Phi = -angular error, F = gamma * Phi(s') - Phi(s), per agent step
            (t0, e0), (t1, e1) = aim0, self._snap["aim"]
            if t0 is not None and t0 == t1 and abs(e0 - e1) < 90.0:
                reward += self.aim_shaping * (e0 - self.shaping_gamma * e1)
        t1, e1 = self._snap["aim"]
        if e1 is not None:
            if self.offtarget_cost:
                reward -= self.offtarget_cost * min(e1, 45.0)
            if self._acquired_t is None and e1 < self.acquire_deg:
                self._acquired_t = s.time - self._ep_start
                if self.acquire_bonus:
                    reward += self.acquire_bonus * max(0.0, 1.0 - self._acquired_t / self.acquire_window)
        if a[4] and self.bullet_cost:
            reward -= self.bullet_cost
        reward -= self.time_penalty
        frame = b.frame()
        self._obs_queue.append(frame if frame is not None else self._obs_queue[-1])
        self._aux_queue.append(self._target(s))
        truncated = (s.time - self._ep_start) >= self.episode_seconds and not terminated
        self._no_contest = False
        if self.no_contest_after and not terminated and self._me(s).alive and self._others_alive(s):
            self._hidden_steps = self._hidden_steps + 1 if s.aux[7] < 0.999 else 0  # aux[7] = this step's LoS trace
            if self._hidden_steps * self.tick * self.repeat >= self.no_contest_after:
                truncated, self._no_contest = True, True
        return self._obs_queue[0], reward, terminated, truncated, self._info(s)

    # ------------------------------------------------------------------ curriculum knobs
    def set_reaction_delay(self, seconds: float):
        """Takes effect at the next reset (training-time curriculum; evaluate at 0.20)."""
        self.delay_steps = int(round(seconds / (self.tick * self.repeat)))

    def set_bot_level(self, level: int):
        """Arena curriculum (next reset): 0 frozen & passive; 1 passive bot strafing left/right along a short path that
        stays in view (we move it ourselves -- a free-roaming bot walks out of sight in ~78% of rounds on dust2);
        2 frozen & shoots back; 3 normal YaPB bot (moves & shoots back)."""
        self.bot_level = int(level)

    def _apply_bot_level(self):
        if self._bot_level_applied == self.bot_level:
            return
        freeze, ignore = {0: (1, 1), 1: (1, 1), 2: (1, 0)}.get(self.bot_level, (0, 0))
        self.bridge.command(f"yb_freeze_bots {freeze}; yb_ignore_enemies {ignore}")
        self._bot_level_applied = self.bot_level

    def _target(self, s):
        """[yaw err, pitch err, in-view, world yaw, world pitch, valid] of the nearest live enemy's chest (degrees).
        The first three are the auxiliary label; the world angles let the teacher label below aim at where the enemy
        was in the frame the agent sees, from where the crosshair points now."""
        lab = self._aux_label(s)
        valid = float(self._me(s).alive and any(k != s.playernum and p.connected and p.alive for k, p in enumerate(s.players)))
        return np.array([lab[0], lab[1], lab[2], lab[0] + s.viewangles[1], lab[1] + s.viewangles[0], valid], np.float32)

    def _aux_ext(self, s):
        """Training-only labels (never observed by the policy), 6 values:
        [0:3] the auxiliary label NOW (the agent sees it `delay_steps` steps late: predicting it = undoing the delay);
        [3:6] teacher action (yaw bin, pitch bin, valid): what the delayed oracle (scratch/ceiling_delay.py, 94-98% at
        every delay) does -- turn toward where the enemy was in the frame the agent sees, from the current view angles.
        Valid only when that frame shows the enemy in view and the agent is alive."""
        q = getattr(self, "_aux_queue", None)
        if not q:
            return np.zeros(6, np.float32)
        past, now = q[0], q[-1]
        out = np.zeros(6, np.float32)
        out[0:3] = now[0:3]
        if past[5] > 0 and past[2] > 0 and self._me(s).alive:
            dy = (float(past[3]) - s.viewangles[1] + 180.0) % 360.0 - 180.0
            dp = float(past[4]) - s.viewangles[0]
            out[3] = int(np.argmin(np.abs(YAW_TABLE - dy)))
            out[4] = int(np.argmin(np.abs(-PITCH_TABLE - dp)))  # the env turns the view pitch by -PITCH_TABLE[a1]
            out[5] = 1.0
        return out

    def _aux_label(self, s):
        """(signed yaw error, signed pitch error, in-view flag) of the nearest live enemy's chest relative
        to the crosshair, degrees. Training-only label for an auxiliary 'where is the enemy' loss."""
        me = s.playernum
        m = self._me(s)
        best = (0.0, 0.0, 0.0, 1e9)
        if m.alive:
            eye = np.array(m.origin) + (0.0, 0.0, 17.0)
            for k, p in enumerate(s.players):
                if k == me or not p.connected or not p.alive:
                    continue
                d = np.array(p.origin) + (0.0, 0.0, 10.0) - eye
                yaw = math.degrees(math.atan2(d[1], d[0]))
                pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
                dy = (yaw - s.viewangles[1] + 180.0) % 360.0 - 180.0
                dp = pitch - s.viewangles[0]
                e = math.hypot(dy, dp)
                if e < best[3]:
                    best = (dy, dp, float(abs(dy) < 45.0 and abs(dp) < 37.0), e)
        return np.array(best[:3], np.float32)

    def _arena_place(self):
        b = self.bridge
        s = b.wait_until(lambda s: self._me(s).alive and self._others_alive(s) and self._bots_ready(s), max_steps=3000)
        P, pairs, dist = self._arena
        ok = np.nonzero(dist <= self.arena_max_dist)[0]
        if self.arena_min_dist > 0 and self.np_random.random() < self.arena_band_frac:
            band = np.nonzero((dist > self.arena_min_dist) & (dist <= self.arena_max_dist))[0]
            if len(band):
                ok = band
        me = s.playernum
        bot = next(k for k, p in enumerate(s.players) if p.connected and k != me)
        self._strafe = None
        self._leash = None
        self._pocket_vis = None
        leash = self.bot_level == 3 and self.arena_leash > 0 and self.np_random.random() < self.arena_leash_frac
        for attempt in range(12):
            q = self.np_random.choice(ok)
            i, j = pairs[q]
            swap = self.np_random.random() < 0.5
            if swap:
                i, j = j, i
            a, e = P[i], P[j]
            if self.bot_level == 3 and (self.arena_min_vis > 0 or leash):
                vis, pts = self._pocket(a, e, key=(int(q), int(swap)))
                self._pocket_vis = vis
                if self.arena_min_vis > 0 and vis < self.arena_min_vis and attempt < 11:
                    continue  # cover right next to the bot's spot: another pair from the same distance band
                if leash:
                    self._leash = dict(bot=bot, c=e.copy(), pts=[e.copy()] + pts, n=0)
            break
        self._ep_dist = float(np.linalg.norm(e - a))
        d = e - a
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))  # engine: positive pitch = down
        off = self.arena_max_offset
        lo = min(self.arena_min_offset, off)
        if lo > 0 and self.np_random.random() < self.arena_review_frac:
            mag = float(self.np_random.uniform(0.0, lo))  # review round: an easier angle (does not count for promotion)
        else:
            mag = float(self.np_random.uniform(lo, off))  # |yaw error| in [min, max]: big-flick stages can't be padded
        yaw += mag if self.np_random.random() < 0.5 else -mag
        pitch = float(np.clip(pitch + self.np_random.uniform(-off, off) / 4, -60, 60))
        self._ep_offset = mag
        if self.bot_level == 1:
            path = self._strafe_path(a, e)
            if path is None:  # no usable strafe path here: try another spawn pair
                self._strafe_retry = getattr(self, "_strafe_retry", 0) + 1
                if self._strafe_retry < 8:
                    return self._arena_place()
            self._strafe_retry = 0
            if path is not None:
                lo, hi, u = path
                spd = float(self.np_random.uniform(120.0, 220.0))
                self._strafe = dict(bot=bot, c=e.copy(), u=u, lo=lo, hi=hi, pos=0.0,
                                    vel=spd * (1 if self.np_random.random() < 0.5 else -1), face=yaw + 180.0,
                                    left=float(self.np_random.uniform(0.3, 0.8)))
        b.step(cmd=f"cs2ai_place {me} {a[0]:.1f} {a[1]:.1f} {a[2]:.1f} {pitch:.2f} {yaw:.2f} 100", capture=False)
        b.step(cmd=f"cs2ai_place {bot} {e[0]:.1f} {e[1]:.1f} {e[2]:.1f} 0 {yaw + 180:.2f} 100", capture=False)
        for _ in range(3):
            s = b.step(capture=False)
        return s

    def _los_cmd(self):
        """World trace from the agent's eye to the nearest live enemy's chest into aux[7] (1 = visible)."""
        s = self.bridge.state
        me = s.playernum
        m = self._me(s)
        best = None
        for k, p in enumerate(s.players):
            if k != me and p.connected and p.alive:
                d = float(np.linalg.norm(np.array(p.origin) - np.array(m.origin)))
                if best is None or d < best[0]:
                    best = (d, p.origin)
        if best is None or not m.alive:
            return "cs2ai_trace 7 0 0 0 0 0 0"
        a, e = np.array(m.origin) + (0, 0, 17), np.array(best[1]) + (0, 0, 10)
        return f"cs2ai_trace 7 {a[0]:.0f} {a[1]:.0f} {a[2]:.0f} {e[0]:.0f} {e[1]:.0f} {e[2]:.0f}"

    def _trace(self, segs):
        """World line traces (<= 3 per bridge step); returns fractions (1 = clear)."""
        out = []
        for k in range(0, len(segs), 3):
            chunk = segs[k:k + 3]
            cmd = ";".join(f"cs2ai_trace {q} {p0[0]:.0f} {p0[1]:.0f} {p0[2]:.0f} {p1[0]:.0f} {p1[1]:.0f} {p1[2]:.0f}"
                           for q, (p0, p1) in enumerate(chunk))
            st = self.bridge.step(cmd=cmd, apply=False, capture=False)
            out += [float(st.aux[q]) for q in range(len(chunk))]
        return out

    def _strafe_path(self, a, e):
        """Pick a strafe segment through the bot's spot, perpendicular to the line of sight, that is free of walls,
        on level floor, and fully visible from the agent's eye. Returns (lo, hi, unit axis) in units, or None."""
        d = (e - a)[:2]
        u = np.array([-d[1], d[0], 0.0]) / (np.linalg.norm(d) + 1e-6)
        want = float(self.np_random.uniform(80.0, 200.0))
        eye = a + (0.0, 0.0, 17.0)
        lim = []
        for sgn in (-1.0, 1.0):
            L = want
            for _ in range(3):
                end = e + sgn * L * u
                wall, floor = self._trace([(e + (0, 0, 10), end + (0, 0, 10)), (end, end - (0, 0, 80))])
                if wall < 0.999:  # wall in the way: stop 24 units before it
                    L = max(0.0, wall * L - 24.0)
                    continue
                if not (0.4 < floor * 80.0 / 36.0 < 1.25):  # floor must stay ~36 units below the hull centre
                    L *= 0.5
                    continue
                vis = self._trace([(eye, end + (0, 0, 10)), (eye, end + (0, 0, 28)), (eye, end - (0, 0, 30))])
                if min(vis) < 0.999:
                    L *= 0.5
                    continue
                break
            else:
                L = 0.0
            lim.append(L)
        if lim[0] + lim[1] < 60.0:
            return None
        return -lim[0], lim[1], u

    def _strafe_tick(self):
        """Advance the scripted strafe by one engine tick; returns the console command that moves the bot."""
        st = self._strafe
        if st is None:
            return ""
        dt = self.tick
        st["pos"] += st["vel"] * dt
        st["left"] -= dt
        if st["pos"] <= st["lo"] or st["pos"] >= st["hi"] or st["left"] <= 0:  # change direction like an AD strafe
            st["pos"] = float(np.clip(st["pos"], st["lo"], st["hi"]))
            st["vel"] = -st["vel"] if st["left"] > 0 or self.np_random.random() < 0.8 else st["vel"]
            if st["pos"] in (st["lo"], st["hi"]):
                st["vel"] = abs(st["vel"]) * (1 if st["pos"] == st["lo"] else -1)
            st["left"] = float(self.np_random.uniform(0.3, 0.8))
        p = st["c"] + st["pos"] * st["u"]
        return f"cs2ai_place {st['bot']} {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} 0 {st['face']:.1f}"

    @staticmethod
    def _pocket_points(e, rot=0.0):
        """The 24 pocket points around spot e: radii 64/128/192, 8 headings each, first heading at `rot`."""
        return [e + np.array([rad * math.cos(rot + k * math.pi / 4), rad * math.sin(rot + k * math.pi / 4), 0.0])
                for rad in (64.0, 128.0, 192.0) for k in range(8)]

    def _pocket_trace(self, a, e, rot=0.0):
        """Trace the pocket in the engine (72 world traces = 24 engine steps). Returns (points, reachable[24],
        visible-and-reachable[24]). A point is REACHABLE if the hull-centre line from e is wall-free and there is floor
        ~36 units below it, VISIBLE if the agent's eye at a sees the chest there."""
        eye = a + (0.0, 0.0, 17.0)
        pts, segs = self._pocket_points(e, rot), []
        for p in pts:
            segs += [(e + (0, 0, 10), p + (0, 0, 10)), (p + (0, 0, 10), p - (0, 0, 80)), (eye, p + (0, 0, 10))]
        fr = self._trace(segs)
        reach = np.zeros(len(pts), bool)
        see = np.zeros(len(pts), bool)
        for q in range(len(pts)):
            wall, floor, vis = fr[3 * q], fr[3 * q + 1], fr[3 * q + 2]
            reach[q] = wall >= 0.999 and 26.0 <= floor * 90.0 <= 58.0  # 90-unit drop trace stops 26-58 units down = level floor
            see[q] = reach[q] and vis >= 0.999
        return pts, reach, see

    def _pocket(self, a, e, key=None):
        """Visibility of the bot's surroundings (24 points around its spot e, see _pocket_trace).
        Returns (visible / reachable, list of visible reachable points within 128 units).
        The map never changes, so scripts/make_arena_pockets.py traces every (pair, orientation) once, with the first
        heading at 0 degrees; with that table (`key` = (pair index, swapped)) this is a lookup instead of 24 engine
        steps per reset attempt. Without the table it traces live, at a random rotation as before."""
        if key is not None and self._pockets is not None:
            mask = int(self._pockets[1][key])
            pts = self._pocket_points(e, 0.0)
            return float(self._pockets[0][key]), [pts[q] for q in range(16) if mask >> q & 1]
        pts, reach, see = self._pocket_trace(a, e, self.np_random.uniform(0.0, math.pi / 4))
        n = int(reach.sum())
        return (float(see.sum()) / n if n else 1.0), [pts[q] for q in range(16) if see[q]]

    def _leash_tick(self):
        """Leashed free-roaming bot: when it has walked further than arena_leash from its spot, put it back on a random
        visible point of the pocket, facing the agent (YaPB keeps its own aim/fire/movement logic)."""
        L = self._leash
        if L is None:
            return ""
        s = self.bridge.state
        bp, m = s.players[L["bot"]], self._me(s)
        if not (bp.alive and m.alive) or np.linalg.norm((np.array(bp.origin) - L["c"])[:2]) <= self.arena_leash:
            return ""
        L["n"] += 1
        p = L["pts"][int(self.np_random.integers(len(L["pts"])))]
        d = np.array(m.origin) - p
        return f"cs2ai_place {L['bot']} {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} 0 {math.degrees(math.atan2(d[1], d[0])):.1f}"

    def render(self):
        f = self.bridge.frame() if self.bridge else None
        return f if f is not None else self._blank

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

    # ------------------------------------------------------------------ reward bookkeeping
    def _aim_error(self, s):
        """(enemy slot, angular error in degrees) from the crosshair to the nearest live enemy's chest."""
        me = s.playernum
        m = self._me(s)
        if not m.alive:
            return None, None
        eye = np.array(m.origin) + (0.0, 0.0, 17.0)
        best = (None, None)
        for k, p in enumerate(s.players):
            if k == me or not p.connected or not p.alive:
                continue
            d = np.array(p.origin) + (0.0, 0.0, 10.0) - eye
            yaw = math.degrees(math.atan2(d[1], d[0]))
            pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
            dy = (yaw - s.viewangles[1] + 180.0) % 360.0 - 180.0
            err = math.hypot(dy, pitch - s.viewangles[0])
            if best[1] is None or err < best[1]:
                best = (k, err)
        return best

    def _snapshot(self, s):
        me = s.playernum
        tgt, err = self._aim_error(s) if (self.aim_shaping or self.offtarget_cost or self.acquire_bonus) else (None, None)
        return {
            "aim": (tgt, err),
            "frags": self._me(s).frags,
            "alive": self._me(s).alive,
            "myhp": self._me(s).health,
            "hp": {i: p.health for i, p in enumerate(s.players) if p.connected and i != me},
            "alive_others": {i: p.alive for i, p in enumerate(s.players) if p.connected and i != me},
        }

    def _account(self, s):
        prev = self._snap
        cur = self._snapshot(s)
        killed = max(0, cur["frags"] - prev["frags"])
        died = prev["alive"] and not cur["alive"]
        # damage proxy: health lost by enemies this tick (exact in 1v1; approximate in FFA,
        # where bots also shoot each other) -- only counted when an enemy died to us or
        # while we are holding fire would be stricter; kept simple and documented.
        dmg = 0
        for i, hp in cur["hp"].items():
            if i in prev["hp"] and prev["alive_others"].get(i) and hp < prev["hp"][i]:
                dmg += prev["hp"][i] - max(hp, 0)
        if self.scn.bots > 1:
            dmg = dmg if killed else 0  # FFA: only credit damage on ticks where we got the frag
        self._kills += killed
        self._deaths += int(died)
        self._damage += dmg
        self._snap = cur
        r = self.rw_kill * killed - self.rw_death * died + self.rw_dmg * dmg
        if self.hurt_penalty and prev["alive"]:
            r -= self.hurt_penalty * max(0, prev["myhp"] - max(cur["myhp"], 0))
        return r, died, killed

    def _info(self, s):
        me = self._me(s)
        return {
            "time": s.time - getattr(self, "_ep_start", s.time),
            "kills": getattr(self, "_kills", 0),
            "deaths": getattr(self, "_deaths", 0),
            "damage": getattr(self, "_damage", 0),
            "health": me.health,
            "alive": bool(me.alive),
            "spawn_dist": self._ep_dist,
            "spawn_offset": self._ep_offset,
            "no_contest": bool(getattr(self, "_no_contest", False)),
            "leashed": bool(self._leash is not None),
            "pocket_vis": self._pocket_vis,
            "acquire_time": self._acquired_t,
            "aux": self._aux_queue[0][:3].copy() if getattr(self, "_aux_queue", None) else np.zeros(3, np.float32),
            "aux_ext": self._aux_ext(s),
        }
