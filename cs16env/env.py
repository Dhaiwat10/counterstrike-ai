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
        if self.scn.arena:
            z = np.load(self.scn.arena)
            self._arena = (z["positions"], z["pairs"], z["dist"])
        self.arena_max_dist = self.scn.arena_max_dist
        # training-only shaping: reward per degree the crosshair moves toward the nearest live enemy
        # (potential-based, Phi = -angular error; computed from server state, never observed by the agent)
        self.aim_shaping = 0.0
        self.bullet_cost = 0.0         # training only: reward per step spent holding fire
        self.time_penalty = 0.0        # training only: reward per agent step until the round ends
        self.hurt_penalty = 0.0        # training only: reward per HP the agent loses
        self.shaping_gamma = 0.99      # potential-based shaping uses gamma * Phi(s') - Phi(s)
        self.episode_seconds = self.scn.episode_seconds  # can grow with the arena distance
        self._ep_dist = 0.0
        self.bot_level = 2             # arena: 0 = frozen & passive, 1 = frozen & shoots, 2 = normal
        self._bot_level_applied = None
        self.arena_max_offset = self.scn.arena_max_offset
        self._obs_queue: collections.deque = collections.deque()
        self._blank = np.zeros((height, width, 3), np.uint8)

    # ------------------------------------------------------------------ engine lifecycle
    def _boot(self):
        self.bridge = EngineBridge(self.cfg)
        b = self.bridge
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
        self._snap = self._snapshot(s)
        self._kills = self._deaths = 0
        self._damage = 0
        frame = b.frame()
        first = frame if frame is not None else self._blank
        self._obs_queue = collections.deque([first] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        # privileged labels aligned with the (delayed) observation, for auxiliary training losses only
        lab = self._aux_label(s)
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
            s = b.step(dyaw=dyaw, dpitch=dpitch, forward=fwd, side=side, buttons=buttons,
                       capture=(i == self.repeat - 1))
            r, died, killed = self._account(s)
            reward += r
            if (died and self.scn.end_on_death) or (killed and self.scn.end_on_kill):
                terminated = True
        if self.aim_shaping:  # potential-based: Phi = -angular error, F = gamma * Phi(s') - Phi(s), per agent step
            (t0, e0), (t1, e1) = aim0, self._snap["aim"]
            if t0 is not None and t0 == t1 and abs(e0 - e1) < 90.0:
                reward += self.aim_shaping * (e0 - self.shaping_gamma * e1)
        if a[4] and self.bullet_cost:
            reward -= self.bullet_cost
        reward -= self.time_penalty
        frame = b.frame()
        self._obs_queue.append(frame if frame is not None else self._obs_queue[-1])
        self._aux_queue.append(self._aux_label(s))
        truncated = (s.time - self._ep_start) >= self.episode_seconds and not terminated
        return self._obs_queue[0], reward, terminated, truncated, self._info(s)

    # ------------------------------------------------------------------ curriculum knobs
    def set_reaction_delay(self, seconds: float):
        """Takes effect at the next reset (training-time curriculum; evaluate at 0.20)."""
        self.delay_steps = int(round(seconds / (self.tick * self.repeat)))

    def set_bot_level(self, level: int):
        """Arena curriculum: 0 frozen & passive, 1 frozen & shoots, 2 moves & shoots (next reset)."""
        self.bot_level = int(level)

    def _apply_bot_level(self):
        if self._bot_level_applied == self.bot_level:
            return
        freeze, ignore = {0: (1, 1), 1: (1, 0)}.get(self.bot_level, (0, 0))
        self.bridge.command(f"yb_freeze_bots {freeze}; yb_ignore_enemies {ignore}")
        self._bot_level_applied = self.bot_level

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
        i, j = pairs[self.np_random.choice(ok)]
        if self.np_random.random() < 0.5:
            i, j = j, i
        a, e = P[i], P[j]
        self._ep_dist = float(np.linalg.norm(e - a))
        d = e - a
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))  # engine: positive pitch = down
        off = self.arena_max_offset
        yaw += self.np_random.uniform(-off, off)
        pitch = float(np.clip(pitch + self.np_random.uniform(-off, off) / 4, -60, 60))
        me = s.playernum
        bot = next(k for k, p in enumerate(s.players) if p.connected and k != me)
        b.step(cmd=f"cs2ai_place {me} {a[0]:.1f} {a[1]:.1f} {a[2]:.1f} {pitch:.2f} {yaw:.2f} 100", capture=False)
        b.step(cmd=f"cs2ai_place {bot} {e[0]:.1f} {e[1]:.1f} {e[2]:.1f} 0 {yaw + 180:.2f} 100", capture=False)
        for _ in range(3):
            s = b.step(capture=False)
        return s

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
        tgt, err = self._aim_error(s) if self.aim_shaping else (None, None)
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
            "aux": self._aux_queue[0] if getattr(self, "_aux_queue", None) else np.zeros(3, np.float32),
        }
