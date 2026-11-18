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
POV demo stats, see docs/PHASE0_RESULTS.md):
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
        # arena, training (2026-11-13): this share of rounds draws only spawn pairs whose enemy has a wall/box within
        # arena_wall_dist units behind it along the line of sight (live trace; up to 12 tries) -- the case where dark/sand
        # outfits were missed most (leet: 66% kills vs 92%). 0 = off.
        self.arena_wall_frac = 0.0
        self.arena_wall_dist = 150.0
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
        self._since_shot = 99  # agent steps since the agent last held fire (for the teacher's tap cadence)
        self._burst = 0        # consecutive agent steps the agent has held fire (for the "reach" fire rule)
        # teacher's fire label (training only, see _fire_label): "tap" = the oracle's rule (default, unchanged);
        # "reach" = on the body with a cadence by distance
        self.fire_rule = "tap"
        # training only: reward per degree of view movement on steps that start with the crosshair within hold_deg of
        # the enemy's chest (0 = off)
        self.hold_cost = 0.0
        self.hold_deg = 1.5
        # navigator teacher (cs16env/nav.py), off unless enable_nav() is called: then every step also traces line of
        # sight to the enemy (aux slots 6/7) and info carries "nav_ext" (see _nav_ext); env.nav_action is its action
        self.nav = None
        self.nav_action = None
        # real duel options (scenarios with reset_round; default off = the game's own spawns and a normal YaPB bot):
        # duel_random_spawn: after the restart the agent is moved to a random spawn point of its own side (nav.MAP_SPAWNS)
        #   (heading: see duel_start_facing below).
        # duel_bot_mode "camp" = the STAYS-PUT SHOOTER: the bot is moved to a random spawn point of its side, facing the
        #   other side (+-camp_yaw_jitter deg), and frozen (yb_freeze_bots 1, which also stops its aim and fire) while the
        #   agent is out of its line of sight; it is unfrozen (normal YaPB: turns, aims, shoots, may strafe) as soon as
        #   there is line of sight either way (eye -> chest or eye -> head trace) or it is hurt, and frozen again after
        #   camp_refreeze_s without line of sight; if it has moved more than camp_home_radius from its spot by then, it is
        #   put back there facing its original direction. It never walks over to the agent.
        # aim_terms_los_only: aim shaping / off-target cost / acquire bonus count only on steps with line of sight
        #   (in a real duel the enemy is behind walls most of the time; these terms were built for the aim arena).
        self.duel_random_spawn = False
        # heading of both players at the start of a real-duel round. "toward" (default since 2026-11-13, the owner's rule:
        # "you are always facing in the general direction of your opponent"): straight across to the other side
        # +-duel_start_jitter deg, also when the game's own spawn point is used (the map's own spawn headings face AWAY
        # from the other side). "any" = the old behaviour: 0-360 deg with duel_random_spawn, the map's heading otherwise.
        self.duel_start_facing = "toward"
        self.duel_start_jitter = 30.0
        # with duel_random_spawn: failure-weighted spawn choice (training only). Each own-side spawn point keeps a running
        # time-out rate (rounds that ran out of time); a spawn is drawn with weight 1 + duel_spawn_adapt x that rate.
        # 0 = uniform (default).
        self.duel_spawn_adapt = 0.0
        self._spawn_rate = None
        self._spawn_last = None
        self._last_trunc = False
        self.duel_bot_mode = "normal"
        self.camp_yaw_jitter = 25.0
        self.camp_refreeze_s = 1.0
        self.camp_home_radius = 64.0
        # duel_bot_mode "side" = MOVES ON ITS OWN SIDE: a normal YaPB bot (walks, peeks, turns, aims, shoots) that is
        #   kept behind its side's cover line (nav.MAP_SPAWNS[map][side]["line_y"]): when it gets within side_margin of
        #   the line and the agent cannot see it, it is moved to a random navigation spot of its own area that the agent's
        #   spot cannot see (never a snap in view); if it reaches that point while the agent CAN see it (mid-fight), it is
        #   frozen where it stands (see side_hard / _side_hold_tick). Its spot at the round start: a random spawn point of its side, facing the other side.
        #   In a fight (line of sight) it is free (normal YaPB) unless it gets within side_hard of the line: then its body
        #   is held there (see _side_hold_tick) until the agent has not seen it for side_hide_s.
        self.side_margin = 60.0
        self.side_hard = 20.0
        self.side_hold_err = 1.5
        self.side_hide_s = 0.5
        self.aim_terms_los_only = False
        # duel_bot_mode "hold" = the HOLD-SPOT opponent (cs16env/peeker.py HoldBot, added 2026-11-13): stands at a spot of
        #   the hold table (cs16env/MAP_hold_spots.json: where the owner stood while in line of sight, mirrored to both
        #   ends, raised spots included) on its own side, faces the agent's side, scripted aim as the peeker, YaPB fires;
        #   changes spot only while unseen (live traces confirm). hold_move_after = (min, max) seconds unseen before a move.
        self.hold_spots_path = None    # None = cs16env/<map>_hold_spots.json (committed; built by scripts/make_hold_spots.py)
        self.hold_move_after = (2.0, 6.0)
        # share of "hold" rounds whose holder NEVER changes spot (training, added 2026-11-13 peek_v6: the exam's holder fixed
        # on the stone steps found the model holding where it could not see it -> 48/100 time-outs; the agent must learn
        # to reposition until it finds the holder). 0 = off.
        self.hold_fixed_frac = 0.0
        # training only (peek_v7): weight multiplier for RAISED hold spots (z > -420: the stone steps) in the hold opponent's
        # and the arena's spot choice; the model still loses to the steps holder (dark outfits) from the CT end. 1 = the
        # owner's own weights (exams).
        self.hold_raised_mult = 1.0
        # training only (peek_v11, diagnosis (a)): share of "hold" rounds whose holder is FIXED on its side's stone steps
        # (the raised table spot with the most weight) -- exactly the exam case the model at T loses. 0 = off.
        self.hold_steps_frac = 0.0
        # training only (peek_v12, go looking): share of "hold" rounds whose holder is FIXED at a spot that sees none of the
        # other side's hold spots (table "sees" lists) -- a holder that never shows up, so the round goes 15 s without
        # contact and the teacher goes looking. 0 = off.
        self.hold_hidden_frac = 0.0
        # THE SWITCH (owner design, 2026-11-13; default OFF). Real duels vs the scripted opponents ("peek", "hold"): while the
        #   agent's origin is more than open_switch_margin units past its OWN cover line (nav.MAP_SPAWNS line_y / toward)
        #   the opponent gets the deadly settings -- peeker.DEADLY_AIM (or open_switch_aim) + pre-aim through walls, the
        #   peeker leaves cover at once and keeps holding while it sees the agent, and the console commands
        #   open_switch_on_cmd (YaPB's own reaction / noticing); back behind the line it returns to its normal stage
        #   settings (open_switch_off_cmd; None = this stage's yb_difficulty and YaPB's defaults). The roaming YaPB bot
        #   ("normal") and the stays-put shooter ("camp") are NOT switched. info["camp"]["switch_s"] = seconds on this round.
        self.duel_open_switch = False
        self.open_switch_margin = 32.0
        self.open_switch_aim = None    # None = peeker.DEADLY_AIM; or ((react lo, hi), turn deg/s, error deg)
        self.open_switch_preaim = True
        # with the switch on, the opponent also fires by script (peeker.trigger -> engine command `cs2ai_fire slot n pitch
        # yaw`, patches/xash3d-lockstep.patch 2026-11-13): the bot presses attack aimed exactly along the scripted view.
        # Needs the rebuilt engine; an older engine ignores the command (YaPB's own trigger only).
        self.open_switch_fire = True
        # height above the spot's standing origin at which the scripted body (peek / hold) is put every tick. 1.0 = the old
        # value, kept for the normal stage settings: it leaves the bot 1 unit in the AIR every tick, so its rifle has the
        # in-air spread (found 2026-11-13: scripted taps did ~7 HP per shot this way, ~37 standing). With the switch on the
        # body is put exactly on its standing origin (0), so the deadly opponent shoots from the ground.
        self.peek_place_dz = 1.0
        self.deadly_place_dz = 0.0
        self.open_switch_on_cmd = "yb_difficulty 4;yb_notice_model 0;yb_whose_your_daddy 1"
        self.open_switch_off_cmd = None
        self._switch_on = False
        self._switch_steps = 0
        self._switch_n = 0
        self._pending_cmd = ""
        # aim arena (scenario with an arena file), default 0 = off: this share of rounds puts the enemy at a hold-table spot
        # (raised ones included) and the agent at an arena position that sees it (live traces, within arena_max_dist)
        self.arena_hold_frac = 0.0
        # ... and of those rounds this share puts the AGENT at a hold spot of the other side (the real-duel distances,
        # 1100-1600 units, that peek_v1 lost at), the rest at any arena position that sees the spot
        self.arena_hold_home = 0.5
        self._arena_hold = False
        # training only, real duels (needs line-of-sight traces): costs on the SEARCH view while no enemy is in line of
        # sight at the step's start (never while one is visible: flicks to a target stay free). search_flip_cost per
        # step-to-step reversal of the turn direction (left then right or right then left). Turn size and sustained turning
        # in one direction cost nothing (a fast 180-degree turn-around is free).
        self.search_flip_cost = 0.0
        self._search_sign = 0
        # training only, real duels with enable_nav(): ROUTE PROGRESS reward per unit of walking distance (along the
        # navigation graph) gained toward the enemy's spawn side, on steps with no enemy in line of sight. It telescopes:
        # walking back costs what walking forward paid, so it cannot be farmed (sum over a round = route_shaping x net
        # progress). still_cost: per step with no enemy in line of sight and horizontal speed < 20 u/s. Both 0 = off.
        self.route_shaping = 0.0
        self.still_cost = 0.0
        # training only, real duels (added 2026-11-13, peek_v3): timeout_penalty = reward added once when a round runs out
        # of time (the per-step time penalty alone made a 45 s time-out cheaper than dying mid-round); los_move_cost = per
        # step x (horizontal speed / 250) on steps that start with the enemy in line of sight (stop to fight: the owner
        # moves ~30 u/s in sight, the model ran 130-140 u/s and wandered in view without shooting). Both 0 = off.
        self.timeout_penalty = 0.0
        self.los_move_cost = 0.0
        # training labels (peek_v9): while the navigator ENGAGES (enemy in line of sight) its look label = the aim teacher's
        # turn toward the enemy, weighted like the other navigator labels (otherwise there is no look label at all in
        # line of sight and the greedy model froze its view ~20 deg off a far enemy: time-outs). Default off.
        self.nav_engage_look = False
        self.teacher_present = False  # --teacher-present: the aim teacher aims at where the enemy IS now (not where it was in the frame)
        # GO LOOKING (owner, 2026-11-13 late): a real duel has no clock -- hold at home first, and after go_look_after s of a
        # round with no line of sight at all, go looking. Then, for the rest of the round: THE SWITCH is turned off and the
        # navigator teacher's goal becomes "side" (walk toward the enemy side; it stops and fights when it sees the enemy).
        # Before that, or once there has been contact, nothing changes. 0 = off (default). info["go_look"] says it happened.
        self.go_look_after = 0.0
        self._go_look = False
        self._contact = False
        self._nav_goal0 = None
        self._route_d = None
        self._camp = None
        self._peeker = None
        self._freeze_applied = None
        # ENEMY PLAYER MODEL (added 2026-11-13). Each team has 4 YaPB classes (nav.TEAM_MODELS); the bot wears one of the 4
        # of the team opposite scn.human_team. bot_skin 0 (default) = old behaviour: the scenario's `yb_quota` adds the bot
        # at boot with YaPB's own random class and it keeps it for the whole game. bot_skin 1..4 = that class; "mixed" = a
        # random class (from its own RNG, so spawn sequences do not change): before each round, with probability
        # skin_change, a new class is drawn (1.0 = every round). We then add the bot ourselves (`yb add`, quota 0).
        # Changing class or difficulty re-adds the bot (~0.5 s of game time). NOTE: `yb_kickall` is not a YaPB command
        # (the console ignores it; `yb kickall` is, and it also lowers the quota), so the old in-run difficulty change
        # never re-added the bot; set_bot(difficulty=...) does it properly.
        self.bot_skin = 0
        self.skin_change = 1.0
        # "mixed" only, training: failure-weighted class choice (like duel_spawn_adapt). Each class keeps a running rate
        # of rounds the agent did not win against it; a new class is drawn with weight 1 + skin_adapt x that rate. 0 = off.
        self.skin_adapt = 0.0
        self._skin_fail = np.zeros(5)
        self._skin_rng = np.random.default_rng(0)
        self._bot_want = None      # (skin, difficulty) wanted for our own bot; None = scenario quota (old behaviour)
        self._bot_applied = None
        self._bot_skin_now = 0     # class (1..4) the bot wears, if we added it; 0 = unknown (YaPB's own choice)
        self._blank = np.zeros((height, width, 3), np.uint8)

    # ------------------------------------------------------------------ enemy player model / difficulty
    @property
    def enemy_team(self):
        return "CT" if self.scn.human_team == "T" else "T"

    @property
    def enemy_model(self):
        """Name of the class the enemy bot wears, or "unknown" (YaPB's own pick at boot)."""
        from .nav import TEAM_MODELS
        return TEAM_MODELS[self.enemy_team][self._bot_skin_now - 1] if self._bot_skin_now else "unknown"

    def set_bot(self, skin=None, difficulty=None):
        """Wanted class (1..4) and/or YaPB difficulty of the enemy bot; applied at the next reset (re-adds the bot)."""
        cur = self._bot_want or (self._bot_skin_now or 0, self.scn.bot_difficulty)
        sk = cur[0] if skin is None else int(skin)
        if not sk:
            sk = int(self._skin_rng.integers(1, 5))
        self._bot_want = (sk, cur[1] if difficulty is None else int(difficulty))

    def _readd_bot(self):
        """Remove the bot(s) and add one ourselves with the wanted class and difficulty, on the team opposite the agent."""
        b = self.bridge
        skin, diff = self._bot_want
        team = 2 if self.scn.human_team == "T" else 1
        b.command("yb_quota 0; yb kickall")
        b.wait_until(lambda s: not any(p.connected for k, p in enumerate(s.players) if k != s.playernum), max_steps=3000)
        b.command(f"yb_difficulty {diff}; yb add {diff} -1 {team} {skin}")
        b.wait_until(lambda s: self._bots_ready(s) and self._others_alive(s), max_steps=3000)
        self._bot_applied = self._bot_want
        self._bot_skin_now = skin
        self._freeze_applied = None  # a fresh bot: re-send the freeze state when it matters

    def _choose_skin(self, seed):
        if seed is not None:
            self._skin_rng = np.random.default_rng(int(seed) + 7919)
        k = self._bot_skin_now
        if k and getattr(self, "_snap", None) is not None and not getattr(self, "_no_contest", False):
            self._skin_fail[k] = 0.9 * self._skin_fail[k] + 0.1 * float(getattr(self, "_kills", 0) == 0)
        if self.bot_skin == "mixed":
            if self._bot_want is None or self._skin_rng.random() < self.skin_change:
                if self.skin_adapt > 0:
                    w = 1.0 + self.skin_adapt * self._skin_fail[1:]
                    self.set_bot(skin=int(self._skin_rng.choice(4, p=w / w.sum())) + 1)
                else:
                    self.set_bot(skin=int(self._skin_rng.integers(1, 5)))
        elif self.bot_skin:
            self.set_bot(skin=int(self.bot_skin))

    def enable_nav(self, graph_path=None, sides=None, **kw):
        """Turn on the scripted navigator teacher and its labels (default off: nothing changes without this call).
        graph_path: data/arena/MAP_nav.npz (default from the scenario's map); sides: {"T": xyz, "CT": xyz} spawn-side
        centres (default: nav.MAP_SIDES[map]); kw go to nav.Navigator (goal_mode, fire_speed, ...)."""
        from . import nav as navmod
        name = self.scn.map.replace("-", "_")
        g = navmod.NavGraph.load(graph_path or f"data/arena/{name}_nav.npz")
        self._nav_sides = sides or navmod.MAP_SIDES.get(self.scn.map)
        self.nav = navmod.Navigator(g, step_s=self.tick * self.repeat, **kw)
        self._nav_los = False

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
        if "mp_weapons_allow_map_placed 0" in self.scn.server_cmds:
            # the map's ground weapons are created at map load, before the setting takes effect: one round restart
            # removes them (otherwise the first round of a session still shows the rifles)
            t_restart = self.bridge.state.time
            b.command("sv_restart 1")
            b.wait_until(lambda s: s.time > t_restart + 1.2 and self._me(s).alive and self._bots_ready(s), max_steps=5000)

    def _me(self, s):
        return s.players[s.playernum] if 0 <= s.playernum < len(s.players) else s.players[0]

    def _others_alive(self, s):
        return all(p.alive for i, p in enumerate(s.players) if p.connected and i != s.playernum)

    def _route_dist(self, pos):
        """Walking distance from pos to the navigator's enemy-side point over the navigation graph (Dijkstra field per
        destination, cached)."""
        nav = self.nav
        if nav is None or nav.enemy_side is None:
            return None
        g = nav.g
        key = tuple(np.round(nav.enemy_side, 1))
        cache = self.__dict__.setdefault("_route_fields", {})
        if key not in cache:
            import heapq
            src = g.nearest(nav.enemy_side)
            dist = np.full(g.n, np.inf)
            dist[src] = float(np.linalg.norm(g.nodes[src][:2] - nav.enemy_side[:2]))
            pq = [(dist[src], src)]
            while pq:
                dk, k = heapq.heappop(pq)
                if dk > dist[k]:
                    continue
                for j, w in g.adj[k]:
                    if dk + w < dist[j]:
                        dist[j] = dk + w
                        heapq.heappush(pq, (dist[j], j))
            cache[key] = dist
        dist = cache[key]
        k = g.nearest(pos)
        if not np.isfinite(dist[k]):
            return None
        return float(dist[k] + np.linalg.norm(g.nodes[k][:2] - pos[:2]))

    def _others_dead(self, s):
        return not any(p.alive for i, p in enumerate(s.players) if p.connected and i != s.playernum)

    def _bots_ready(self, s):
        n = sum(1 for i, p in enumerate(s.players) if p.connected and i != s.playernum)
        return n >= self.scn.bots

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._choose_skin(seed)
        booted = self.bridge is None
        if booted:
            self._boot()
        if self._bot_want is not None and self._bot_want != self._bot_applied:
            self._readd_bot()  # enemy class / difficulty change (or the first round with our own bot)
        if not booted and self.scn.reset_round:
            # fresh round: both sides respawn at their spawn points
            t_restart = self.bridge.state.time
            if self.scn.restart_confirm:
                # mark the agent with an impossible health (255, in place); the restart respawns it with exactly 100,
                # so health == 100 confirms the round has really restarted (fallback: 3 s)
                s0 = self.bridge.state
                me_i, o = s0.playernum, self._me(s0).origin
                if self._me(s0).alive:
                    self.bridge.command(f"cs2ai_place {me_i} {o[0]:.1f} {o[1]:.1f} {o[2]:.1f} "
                                        f"{s0.viewangles[0]:.2f} {s0.viewangles[1]:.2f} 255")
                self.bridge.command("sv_restart 1")
                self.bridge.wait_until(lambda s: s.time > t_restart + 0.9 and self._me(s).alive
                                       and (self._me(s).health == 100 or s.time > t_restart + 3.0)
                                       and self._bots_ready(s) and self._others_alive(s), max_steps=5000)
            else:
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
        self._camp = None
        self._peeker = None
        self._switch_steps = self._switch_n = 0
        self._go_look, self._contact = False, False
        if self.nav is not None and self._nav_goal0 is not None:
            self.nav.goal_mode = self._nav_goal0  # back to the round's own goal (home) after a go-looking round
        if self._switch_on:  # the previous round ended with the switch on: back to the normal settings first
            self._switch_on = False
            self._pending_cmd = ""
            b.command(self._switch_off_cmd())
        if self.scn.reset_round and (self.duel_random_spawn or self.duel_bot_mode in ("camp", "side", "peek", "hold") or self._freeze_applied
                                     or self.duel_start_facing == "toward"):
            s = self._duel_setup()
        s = b.step(apply=True, capture=True)  # one neutral step to get a fresh frame
        self._ep_start = s.time
        self._since_shot = 99
        self._burst = 0
        self._acquired_t = None
        self._hidden_steps = 0
        self._step_los = False
        self._search_sign = 0
        self._route_d = None
        self._snap = self._snapshot(s)
        self._kills = self._deaths = 0
        self._damage = 0
        frame = b.frame()
        first = frame if frame is not None else self._blank
        self._obs_queue = collections.deque([first] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        # privileged labels aligned with the (delayed) observation, for auxiliary training losses only
        lab = self._target(s)
        self._aux_queue = collections.deque([lab] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        if self.nav is not None:
            self.nav.reset()
            m = self._me(s)
            if self._nav_sides is not None:  # the enemy side = the spawn side the agent did not start on
                dT = np.linalg.norm(np.array(m.origin) - self._nav_sides["T"])
                dC = np.linalg.norm(np.array(m.origin) - self._nav_sides["CT"])
                self.nav.enemy_side = np.array(self._nav_sides["CT" if dT < dC else "T"], np.float64)
            if self.nav.goal_mode == "home":  # home teacher: a hold spot on the agent's OWN side (nav.Navigator.set_home)
                from .nav import MAP_SPAWNS
                from .peeker import side_spots
                L = MAP_SPAWNS[self.scn.map][self.scn.human_team]
                tab = self._hold_table()
                mine = [sp for sp in tab if sp["side"] == self.scn.human_team
                        and (sp["y"] - L["line_y"]) * L["toward"] <= 0]
                P = np.array([(sp["x"], sp["y"], sp["z"]) for sp in mine], np.float64)
                W = np.array([sp["w"] for sp in mine], np.float64)
                lists = None
                if all("sees" in sp for sp in mine):  # real visibility between hold spots (scripts/hold_spot_vis.py)
                    lists = [(np.array([(tab[j]["x"], tab[j]["y"], tab[j]["z"]) for j in sp["sees"]], np.float64).reshape(-1, 3),
                              np.array([tab[j]["w"] for j in sp["sees"]], np.float64)) for sp in mine]
                self.nav.set_home(P, W, self.nav.enemy_side, self.np_random, watch_lists=lists)
            s = b.step(cmd=self._los_cmds(), apply=False, capture=False)
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
        if self.hold_cost and aim0[1] is not None and aim0[1] < self.hold_deg:
            reward -= self.hold_cost * (abs(yaw) + abs(pitch))
        los_traces = self.nav is not None or self._camp is not None or self.aim_terms_los_only
        los_prev = getattr(self, "_step_los", False)
        if self.search_flip_cost and los_traces:
            if not los_prev:
                sg = int(np.sign(yaw))
                if sg and self._search_sign and sg != self._search_sign:
                    reward -= self.search_flip_cost
                if sg:
                    self._search_sign = sg
            else:
                self._search_sign = 0
        for i in range(self.repeat):
            cmds = [self._strafe_tick(), self._leash_tick(), self._side_hold_tick(), self._peek_tick(),
                    self._los_cmd() if i == self.repeat - 1 and self.no_contest_after and not los_traces else "",
                    self._los_cmds() if i == self.repeat - 1 and los_traces else ""]
            cmd = ";".join(c for c in cmds if c)
            if self._pending_cmd and len(cmd) + len(self._pending_cmd) + 1 <= 255:  # the switch's console commands
                cmd = self._pending_cmd + (";" + cmd if cmd else "")
                self._pending_cmd = ""
            s = b.step(dyaw=dyaw, dpitch=dpitch, forward=fwd, side=side, buttons=buttons,
                       capture=(i == self.repeat - 1), cmd=cmd)
            r, died, killed = self._account(s)
            reward += r
            if (died and self.scn.end_on_death) or (killed and self.scn.end_on_kill):
                terminated = True
        if (self.route_shaping or self.still_cost) and self.nav is not None and self._me(s).alive:
            d = self._route_dist(np.array(self._me(s).origin))
            if not los_prev and self._route_d is not None and d is not None:
                reward += self.route_shaping * float(np.clip(self._route_d - d, -30.0, 30.0))
            self._route_d = d
            if not los_prev and math.hypot(*s.velocity[:2]) < 20.0:
                reward -= self.still_cost
        los_now = False
        if los_traces:
            los_now = bool(self._me(s).alive and not self._others_dead(s) and (s.aux[7] >= 0.999 or s.aux[6] >= 0.999))
            self._step_los = los_now
            if not terminated:
                self._camp_tick(s, los_now)
                self._go_look_tick(s, los_now)
                self._switch_tick(s)
        aim_ok = (los_prev and los_now) if self.aim_terms_los_only else True
        if self.aim_shaping and aim_ok:  # potential-based: Phi = -angular error, F = gamma * Phi(s') - Phi(s), per agent step
            (t0, e0), (t1, e1) = aim0, self._snap["aim"]
            if t0 is not None and t0 == t1 and abs(e0 - e1) < 90.0:
                reward += self.aim_shaping * (e0 - self.shaping_gamma * e1)
        t1, e1 = self._snap["aim"]
        if self.aim_terms_los_only and not los_now:
            e1 = None
        if e1 is not None:
            if self.offtarget_cost:
                reward -= self.offtarget_cost * min(e1, 45.0)
            if self._acquired_t is None and e1 < self.acquire_deg:
                self._acquired_t = s.time - self._ep_start
                if self.acquire_bonus:
                    reward += self.acquire_bonus * max(0.0, 1.0 - self._acquired_t / self.acquire_window)
        if a[4] and self.bullet_cost:
            reward -= self.bullet_cost
        self._since_shot = 0 if a[4] else self._since_shot + 1
        self._burst = self._burst + 1 if a[4] else 0
        reward -= self.time_penalty
        frame = b.frame()
        self._obs_queue.append(frame if frame is not None else self._obs_queue[-1])
        self._aux_queue.append(self._target(s))
        truncated = (s.time - self._ep_start) >= self.episode_seconds and not terminated
        if truncated and self.timeout_penalty and self.scn.reset_round:
            reward -= self.timeout_penalty
        if self.los_move_cost and los_prev and self._me(s).alive:
            reward -= self.los_move_cost * min(1.0, math.hypot(*s.velocity[:2]) / 250.0)
        self._last_trunc = bool(truncated)
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

    # ------------------------------------------------------------------ real duel: spawns and the stays-put shooter
    def _set_freeze(self, on: bool):
        if self._freeze_applied != bool(on):
            self.bridge.command(f"yb_freeze_bots {int(bool(on))}")
            self._freeze_applied = bool(on)

    def _duel_setup(self):
        """After the round restart: optional random own-side spawn + heading for the agent; the camping bot's spot."""
        from .nav import MAP_SPAWNS
        b = self.bridge
        s = b.state
        sp = MAP_SPAWNS.get(self.scn.map)
        me = s.playernum
        bot = next((k for k, p in enumerate(s.players) if p.connected and k != me), None)
        self._peeker = None
        if self.duel_bot_mode in ("peek", "hold") and bot is not None and sp is not None:
            self._set_freeze(False)
            return self._peek_setup(s, sp, bot, cmds_mine=self._random_spawn_cmd(sp, me))
        camp = self.duel_bot_mode in ("camp", "side") and bot is not None and sp is not None
        self._set_freeze(camp and self.duel_bot_mode == "camp")  # normal mode: make sure an earlier camping round left no frozen bot behind
        if sp is None:
            return s
        mine = "T" if self.scn.human_team == "T" else "CT"
        theirs = "CT" if mine == "T" else "T"
        cmds = []
        cmds += self._random_spawn_cmd(sp, me)
        if not camp and bot is not None and self.duel_start_facing == "toward":
            # the roaming bot (or the human opponent of train.duel) keeps the game's spawn point, turned to face our side
            o = s.players[bot].origin
            cmds.append(f"cs2ai_place {bot} {o[0]:.1f} {o[1]:.1f} {o[2]:.1f} 0 {self._start_yaw(sp, theirs):.2f}")
        if camp:
            p = sp[theirs]["pts"][int(self.np_random.integers(len(sp[theirs]["pts"])))]
            # facing the agent's side: the map's T spawns face away from CT (yaw 90), so CT faces the T side at 90 deg
            yaw = (sp[theirs]["yaw"] + 180.0 + float(self.np_random.uniform(-self.camp_yaw_jitter, self.camp_yaw_jitter))) % 360.0
            cmds.append(f"cs2ai_place {bot} {p[0]:.1f} {p[1]:.1f} {p[2] + 1:.1f} 0 {yaw:.2f} 100")
            self._camp = dict(bot=bot, home=np.array(p, np.float64) + (0, 0, 1), yaw=yaw, frozen=self.duel_bot_mode == "camp",
                              unseen=0.0, hp=100, homes=0, unfreezes=0, mode=self.duel_bot_mode, side=theirs, line_freezes=0,
                              max_adv=-1e9, crossed=False, walked=0.0, last=np.array(p, np.float64))
        for c in cmds:
            b.step(cmd=c, capture=False)
        for _ in range(3):
            s = b.step(capture=False)
        return s

    def _pick_spawn(self, pts):
        n = len(pts)
        if self._spawn_rate is None or len(self._spawn_rate) != n:
            self._spawn_rate = np.zeros(n)
        if self._spawn_last is not None:  # the previous round's outcome updates its spawn's time-out rate
            k = self._spawn_last
            self._spawn_rate[k] = 0.9 * self._spawn_rate[k] + 0.1 * float(self._last_trunc)
        if self.duel_spawn_adapt > 0:
            w = 1.0 + self.duel_spawn_adapt * self._spawn_rate
            k = int(self.np_random.choice(n, p=w / w.sum()))
        else:
            k = int(self.np_random.integers(n))
        self._spawn_last = k
        return pts[k]

    def _start_yaw(self, sp, side):
        """Heading of a player of `side` at the round start (see duel_start_facing)."""
        if self.duel_start_facing != "toward":
            return float(self.np_random.uniform(0.0, 360.0))
        j = float(self.duel_start_jitter)
        # the map's own spawn heading faces away from the other side (nav.MAP_SPAWNS), so +180 = straight across
        return (sp[side]["yaw"] + 180.0 + float(self.np_random.uniform(-j, j))) % 360.0

    def _random_spawn_cmd(self, sp, me):
        mine = "T" if self.scn.human_team == "T" else "CT"
        if not self.duel_random_spawn:
            if self.duel_start_facing != "toward":
                return []
            o = self.bridge.state.players[me].origin  # the game's own spawn point, turned to face the other side
            return [f"cs2ai_place {me} {o[0]:.1f} {o[1]:.1f} {o[2]:.1f} 0 {self._start_yaw(sp, mine):.2f}"]
        p = self._pick_spawn(sp[mine]["pts"])
        return [f"cs2ai_place {me} {p[0]:.1f} {p[1]:.1f} {p[2] + 1:.1f} 0 {self._start_yaw(sp, mine):.2f} 100"]

    def _peek_setup(self, s, sp, bot, cmds_mine):
        """duel_bot_mode "peek": the scripted peeker body (cs16env/peeker.py) on the bot's side; YaPB decides when to fire."""
        from .nav import NavGraph
        from .peeker import Peeker
        b = self.bridge
        for c in cmds_mine:
            b.step(cmd=c, capture=False)
        s = b.step(capture=False)
        mine = "T" if self.scn.human_team == "T" else "CT"
        theirs = "CT" if mine == "T" else "T"
        if not hasattr(self, "_side_graph"):
            self._side_graph = self.nav.g if self.nav is not None else NavGraph.load(
                f"data/arena/{self.scn.map.replace('-', '_')}_nav.npz")
            self._side_nodes = {}
        L = sp[theirs]
        from .nav import MAP_SIDES
        if self.duel_bot_mode == "hold":
            from .peeker import HoldBot, load_hold_spots, side_spots
            P, W = side_spots(self._hold_table(), theirs, L["line_y"], L["toward"])
            W = W * np.where(P[:, 2] > -420.0, self.hold_raised_mult, 1.0)
            fixed = self.hold_fixed_frac > 0 and self.np_random.random() < self.hold_fixed_frac
            if self.hold_hidden_frac > 0 and self.np_random.random() < self.hold_hidden_frac:
                tab = [sp for sp in self._hold_table() if sp["side"] == theirs and "sees" in sp
                       and (sp["y"] - L["line_y"]) * L["toward"] <= 0]
                if tab:
                    sp = min(tab, key=lambda q: (len(q["sees"]), -q["w"]))
                    P, W, fixed = np.array([[sp["x"], sp["y"], sp["z"]]], np.float64), np.array([1.0]), True
            elif self.hold_steps_frac > 0 and self.np_random.random() < self.hold_steps_frac:
                raised = np.nonzero(P[:, 2] > -420.0)[0]
                k = int(raised[np.argmax(W[raised])])
                P, W, fixed = P[k:k + 1], W[k:k + 1], True
            pk = HoldBot(P, W, L["line_y"], L["toward"], self.np_random, difficulty=self.scn.bot_difficulty,
                         look_point=MAP_SIDES[self.scn.map][mine], move_after=(1e9, 1e9) if fixed else self.hold_move_after)
        else:
            pk = Peeker(self._side_graph, L["line_y"], L["toward"], self.np_random, difficulty=self.scn.bot_difficulty,
                        look_point=MAP_SIDES[self.scn.map][mine])
        p = pk.reset(np.array(self._me(s).origin))
        self._peeker = pk
        self._camp = dict(bot=bot, mode="peek", kind=self.duel_bot_mode, hp=100, unfreezes=0, homes=0, vis_steps=0, live_steps=0,
                          vis_runs=0, idle_vis=0.0, idle_vis_max=0.0, was_vis=False)
        b.step(cmd=f"cs2ai_place {bot} {p[0]:.1f} {p[1]:.1f} {p[2] + 1:.1f} {pk.pitch:.2f} {pk.yaw:.2f} 100", capture=False)
        for _ in range(3):
            s = b.step(capture=False)
        return s

    def _peek_tick(self):
        pk = getattr(self, "_peeker", None)
        if pk is None:
            return ""
        c = self._camp
        s = self.bridge.state
        bp, m = s.players[c["bot"]], self._me(s)
        if not (bp.alive and m.alive):
            return ""
        chest = np.array(m.origin, np.float64) + (0.0, 0.0, 10.0)
        extra = ""
        if pk.phase == "jumpcheck":
            if pk.jump_traced:  # the traces issued last tick: aux[2..5] (1 = clear = visible)
                pk.jump_verdict = all(float(s.aux[k]) < 0.999 for k in (2, 3, 4, 5))
            else:
                eye = np.array(m.origin, np.float64) + (0.0, 0.0, 17.0)
                cur, dst = np.array(bp.origin, np.float64), pk.jump_target()
                segs = [(cur, 10.0), (cur, 28.0), (dst, 10.0), (dst, 28.0)]
                extra = ";" + ";".join(f"cs2ai_trace {2 + q} {eye[0]:.0f} {eye[1]:.0f} {eye[2]:.0f} {pt[0]:.0f} {pt[1]:.0f} "
                                       f"{pt[2] + dz:.0f}" for q, (pt, dz) in enumerate(segs))
                pk.jump_traced = True
        los = bool(getattr(self, "_step_los", False))
        p, pitch, yaw, _ = pk.tick(self.tick, np.array(m.origin, np.float64), chest, los)
        fire = ""
        if self.open_switch_fire and pk.deadly:
            # the shot's own line (bot eye -> its aim point) must be clear of walls/boxes: traced every tick into aux[1]
            # (read on the next tick); the agent's line-of-sight traces alone let it fire into a box edge (found 2026-11-13)
            clear = c.get("aim_traced", False) and float(s.aux[1]) >= 0.999
            if pk.trigger(self.tick, los and clear):
                fire = f";cs2ai_fire {c['bot']} 2 {pitch:.2f} {yaw:.2f}"
                c["fire_ticks"] = c.get("fire_ticks", 0) + 1
            if extra:  # a jump check's 4 traces go out this tick: skip the aim trace (255-character command limit)
                c["aim_traced"] = False
            else:
                e0 = p + (0.0, 0.0, 17.0)
                e1 = chest + (0.0, 0.0, pk.head_dz)
                fire += f";cs2ai_trace 1 {e0[0]:.0f} {e0[1]:.0f} {e0[2]:.0f} {e1[0]:.0f} {e1[1]:.0f} {e1[2]:.0f}"
                c["aim_traced"] = True
        else:
            c["aim_traced"] = False
        return f"cs2ai_place {c['bot']} {p[0]:.1f} {p[1]:.1f} {p[2] + (self.deadly_place_dz if pk.deadly else self.peek_place_dz):.1f} {pitch:.2f} {yaw:.2f}" + fire + extra

    def _hold_table(self):
        if getattr(self, "_hold_cache", None) is None:
            from .peeker import load_hold_spots
            self._hold_cache = load_hold_spots(self.hold_spots_path or str(Path(__file__).parent / f"{self.scn.map.replace('-', '_')}_hold_spots.json"))
        return self._hold_cache

    def _switch_off_cmd(self):
        if self.open_switch_off_cmd is not None:
            return self.open_switch_off_cmd
        return f"yb_difficulty {self.scn.bot_difficulty};yb_notice_model 1;yb_whose_your_daddy 0"

    def _go_look_tick(self, s, los):
        """GO LOOKING (see go_look_after): once per agent step."""
        self._contact |= bool(los)
        if not self.go_look_after or self._go_look or self._contact or not self.scn.reset_round:
            return
        if s.time - self._ep_start < self.go_look_after:
            return
        self._go_look = True
        pk = getattr(self, "_peeker", None)
        if self._switch_on and pk is not None:  # the switch goes OFF for the rest of the round
            self._switch_on = False
            pk.set_deadly(False)
            self._pending_cmd = self._switch_off_cmd()
        if self.nav is not None:
            if self._nav_goal0 is None:
                self._nav_goal0 = self.nav.goal_mode
            self.nav.goal_mode = "side"
            self.nav.t_plan = -1e9  # re-plan toward the enemy side now

    def _switch_tick(self, s):
        """Once per agent step, real duel with duel_open_switch: switch the scripted opponent's deadly settings on while the
        agent is more than open_switch_margin past its own cover line, off when it is back."""
        pk = getattr(self, "_peeker", None)
        if not self.duel_open_switch or pk is None or not self._me(s).alive or self._go_look:
            return
        from .nav import MAP_SPAWNS
        L = MAP_SPAWNS[self.scn.map][self.scn.human_team]
        adv = (float(self._me(s).origin[1]) - L["line_y"]) * L["toward"]
        on = adv > self.open_switch_margin
        if on != self._switch_on:
            self._switch_on = on
            self._switch_n += int(on)
            pk.set_deadly(on, aim=self.open_switch_aim, preaim=self.open_switch_preaim)
            self._pending_cmd = self.open_switch_on_cmd if on else self._switch_off_cmd()
        self._switch_steps += int(on)

    def _peek_step_stats(self, los):
        c = self._camp
        if c is None or c.get("mode") != "peek":
            return
        c["live_steps"] += 1
        c["vis_steps"] += int(los)
        nj = self._peeker.stats["jumps"]
        if nj > c.get("jumps_last", 0) and (los or c["was_vis"]):  # a jump during a step seen before or after it
            c["jumps_seen"] = c.get("jumps_seen", 0) + 1
        c["jumps_last"] = nj
        if los and not c["was_vis"]:
            c["vis_runs"] += 1
        c["was_vis"] = los
        dt = self.tick * self.repeat
        moving = self._peeker.phase in ("out", "back", "walk")
        c["idle_vis"] = c["idle_vis"] + dt if (los and not moving) else 0.0
        c["idle_vis_max"] = max(c["idle_vis_max"], c["idle_vis"])

    def _camp_tick(self, s, los):
        """Once per agent step (after the line-of-sight traces): freeze / unfreeze / send home the camping bot."""
        c = self._camp
        if c is None:
            return
        bp = s.players[c["bot"]]
        if not bp.alive or not self._me(s).alive:
            return
        if c["mode"] == "side":
            return self._side_tick(s, los, c, bp)
        if c["mode"] == "peek":
            return self._peek_step_stats(los)
        hurt = bp.health < c["hp"]
        c["hp"] = bp.health
        if los or hurt:
            c["unseen"] = 0.0
            if c["frozen"]:
                self._set_freeze(False)
                c["frozen"] = False
                c["unfreezes"] += 1
            return
        if c["frozen"]:
            return
        c["unseen"] += self.tick * self.repeat
        if c["unseen"] >= self.camp_refreeze_s:
            self._set_freeze(True)
            c["frozen"] = True
            if np.linalg.norm((np.array(bp.origin) - c["home"])[:2]) > self.camp_home_radius:
                h = c["home"]
                self.bridge.command(f"cs2ai_place {c['bot']} {h[0]:.1f} {h[1]:.1f} {h[2]:.1f} 0 {c['yaw']:.2f}")
                c["homes"] += 1

    def _side_spot(self, s, c):
        """A random navigation spot of the bot's own area (behind its line by >= 100 units) that the agent's nearest spot
        cannot see (precomputed node-to-node visibility); None if there is none."""
        from .nav import MAP_SPAWNS
        L = MAP_SPAWNS[self.scn.map][c["side"]]
        if not hasattr(self, "_side_nodes"):
            from .nav import NavGraph
            g = self.nav.g if self.nav is not None else NavGraph.load(f"data/arena/{self.scn.map.replace('-', '_')}_nav.npz")
            self._side_graph = g
            self._side_nodes = {}
        g = self._side_graph
        key = c["side"]
        if key not in self._side_nodes:
            adv = (g.nodes[:, 1] - L["line_y"]) * L["toward"]
            self._side_nodes[key] = np.nonzero(adv <= -100.0)[0]
        cand = self._side_nodes[key]
        here = g.nearest(np.array(self._me(s).origin))
        ok = [int(k) for k in cand if not g.vis[here, k]]
        if not ok:
            return None
        return g.nodes[ok[int(self.np_random.integers(len(ok)))]]

    def _side_tick(self, s, los, c, bp):
        from .nav import MAP_SPAWNS
        L = MAP_SPAWNS[self.scn.map][c["side"]]
        o = np.array(bp.origin, np.float64)
        adv = (o[1] - L["line_y"]) * L["toward"]  # > 0 = past the cover line
        c["max_adv"] = max(c["max_adv"], adv)
        c["crossed"] |= adv > 0
        c["walked"] += float(np.linalg.norm(o[:2] - c["last"][:2]))
        c["last"] = o
        c["unseen"] = 0.0 if los else c["unseen"] + self.tick * self.repeat
        near = adv > -self.side_margin
        if c.get("hold") is not None:  # held at its limit mid-fight (see _side_hold_tick)
            if c["unseen"] >= self.side_hide_s and self._side_hide(s, c):
                c["hold"] = None
            return
        if near and not los:
            self._side_hide(s, c)

    def _side_hold_tick(self):
        """Every engine tick, real duel vs the "side" bot: once the bot is within side_hard of its cover line while the
        agent can see it, its body is held at that spot (put back every tick, velocity zero: it stops, no jump), facing
        the agent's chest with a fixed random error (side_hold_err deg) chosen when the hold starts; between holds its
        aim, fire and movement are YaPB's own. It is NOT frozen, so it keeps deciding to fire by itself."""
        c = self._camp
        if c is None or c.get("mode") != "side":
            return ""
        s = self.bridge.state
        bp, m = s.players[c["bot"]], self._me(s)
        if not (bp.alive and m.alive):
            return ""
        from .nav import MAP_SPAWNS
        L = MAP_SPAWNS[self.scn.map][c["side"]]
        o = np.array(bp.origin, np.float64)
        if c.get("hold") is None:
            adv = (o[1] - L["line_y"]) * L["toward"]
            if not (getattr(self, "_step_los", False) and adv > -self.side_hard):
                return ""
            e = self.side_hold_err
            c["hold"] = (o.copy(), float(self.np_random.uniform(-e, e)), float(self.np_random.uniform(-e, e) / 2))
            c["line_freezes"] += 1
        p, ey, ep = c["hold"]
        d = np.array(m.origin) + (0.0, 0.0, 10.0) - (p + (0.0, 0.0, 17.0))
        yaw = math.degrees(math.atan2(d[1], d[0])) + ey
        pitch = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))) + ep
        return f"cs2ai_place {c['bot']} {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} {pitch:.2f} {yaw:.2f}"

    def _side_hide(self, s, c):
        """Move the bot to a hidden spot of its own area (only called while the agent cannot see it)."""
        p = self._side_spot(s, c)
        if p is None:
            return False
        self.bridge.command(f"cs2ai_place {c['bot']} {p[0]:.1f} {p[1]:.1f} {p[2] + 1:.1f} 0 {c['yaw']:.2f}")
        c["last"] = np.array(p, np.float64)
        c["homes"] += 1
        return True

    def _target(self, s):
        """[yaw err, pitch err, in-view, world yaw, world pitch, valid] of the nearest live enemy's chest (degrees).
        The first three are the auxiliary label; the world angles let the teacher label below aim at where the enemy
        was in the frame the agent sees, from where the crosshair points now."""
        lab = self._aux_label(s, with_dist=True)
        valid = float(self._me(s).alive and any(k != s.playernum and p.connected and p.alive for k, p in enumerate(s.players)))
        return np.array([lab[0], lab[1], lab[2], lab[0] + s.viewangles[1], lab[1] + s.viewangles[0], valid, lab[3]], np.float32)

    def _aux_ext(self, s):
        """Training-only labels (never observed by the policy), 10 values:
        [0:3] the auxiliary label NOW (the agent sees it `delay_steps` steps late: predicting it = undoing the delay);
        [3:6] teacher action (yaw bin, pitch bin, valid): what the delayed oracle (scripts/ceiling_delay.py, 94-98% at
        every delay) does -- turn toward where the enemy was in the frame the agent sees, from the current view angles.
        Valid only when that frame shows the enemy in view and the agent is alive.
        [6:10] the same teacher in full: the exact turn it wants (yaw degrees, view-pitch degrees), whether it fires
        now (crosshair within the target's angular size and no shot in the agent's last 3 steps: the oracle's tap), and
        a second valid flag that does NOT need the enemy in view. Like the oracle, this teacher always answers while
        both players are alive: with the enemy off screen it says "turn back toward where it was", which is the only
        supervision for recovering after an overshoot. The bins [3:5] are filled whenever this flag is set."""
        q = getattr(self, "_aux_queue", None)
        if not q:
            return np.zeros(10, np.float32)
        past, now = q[0], q[-1]
        if self.teacher_present:  # the true present: where the enemy is now, from where the crosshair points now
            past = now
        out = np.zeros(10, np.float32)
        out[0:3] = now[0:3]
        if past[5] > 0 and self._me(s).alive:
            dy = (float(past[3]) - s.viewangles[1] + 180.0) % 360.0 - 180.0
            dp = float(past[4]) - s.viewangles[0]
            out[3] = int(np.argmin(np.abs(YAW_TABLE - dy)))
            out[4] = int(np.argmin(np.abs(-PITCH_TABLE - dp)))  # the env turns the view pitch by -PITCH_TABLE[a1]
            out[5] = float(past[2] > 0)
            out[6], out[7] = dy, dp
            out[8] = self._fire_label(dy, dp, float(past[6]))
            out[9] = 1.0
        return out

    def _fire_label(self, dy, dp, dist):
        """Teacher's fire decision for the next step, from the crosshair's offset (dy, dp degrees) to the enemy's chest
        (by the real view angle, which ignores the recoil kick; see the recoil notes in docs/PHASE3_NOTES.md 2026-11-12).
        tap  : within ~9 units of the chest (>= 0.4 deg) and no shot in the last 3 steps (the oracle's single taps).
        reach: on the body -- within 10 units sideways and 14 units up/down of the chest point, plus 0.15 deg -- with a
               cadence by distance: under 500 units bursts of up to 5 steps (then 2 steps' pause), 500-1000 units
               bursts of up to 2 steps then 3 steps' pause, beyond 1000 units single taps 3 steps apart (the kick has
               decayed by then, so every shot leaves from where the crosshair is)."""
        d = max(dist, 1.0)
        if self.fire_rule == "reach":
            on = (abs(dy) < math.degrees(math.atan2(10.0, d)) + 0.15) and (abs(dp) < math.degrees(math.atan2(14.0, d)) + 0.15)
            if dist < 500.0:
                ok = (0 < self._burst < 5) or self._since_shot >= 2
            elif dist < 1000.0:
                ok = (0 < self._burst < 2) or self._since_shot >= 3
            else:
                ok = self._since_shot >= 3
            return float(on and ok)
        on = math.hypot(dy, dp) < max(0.4, math.degrees(math.atan2(9.0, d)))
        return float(on and self._since_shot >= 3)

    def _aux_label(self, s, with_dist=False):
        """(signed yaw error, signed pitch error, in-view flag) of the nearest live enemy's chest relative
        to the crosshair, degrees. Training-only label for an auxiliary 'where is the enemy' loss.
        with_dist: also its distance in units, as a 4th value."""
        me = s.playernum
        m = self._me(s)
        best = (0.0, 0.0, 0.0, 1e9, 0.0)
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
                    best = (dy, dp, float(abs(dy) < 45.0 and abs(dp) < 37.0), e, float(np.linalg.norm(d)))
        return np.array(best[:3] + ((best[4],) if with_dist else ()), np.float32)

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
        wall_round = self.arena_wall_frac > 0 and self.np_random.random() < self.arena_wall_frac
        self._arena_hold = False
        if self.arena_hold_frac > 0 and self.np_random.random() < self.arena_hold_frac:
            ae = self._arena_hold_pair(P)
            if ae is not None:
                a, e = ae
                self._arena_hold = True
        for attempt in range(0 if self._arena_hold else 12):
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
            if wall_round and attempt < 11:
                eye = a + (0.0, 0.0, 17.0)
                u = (e + (0.0, 0.0, 10.0)) - eye
                u = u / max(1e-6, float(np.linalg.norm(u)))
                p0 = e + (0.0, 0.0, 10.0) + 24.0 * u
                if self._trace([(p0, p0 + self.arena_wall_dist * u)])[0] >= 0.999:
                    continue  # open behind the enemy: try another pair
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

    def _arena_hold_pair(self, P, tries=12):
        """Aim arena, arena_hold_frac rounds: (agent position, enemy position) with the enemy at a hold-table spot (weighted
        by how long the owner stood there; raised spots included) and the agent at an arena position within
        [max(200, arena_min_dist), arena_max_dist] that sees the enemy's chest AND head (live traces). None if not found."""
        spots = self._hold_table()
        w = np.array([sp["w"] * (self.hold_raised_mult if sp["z"] > -420.0 else 1.0) for sp in spots], np.float64)
        home = self.arena_hold_home > 0 and self.np_random.random() < self.arena_hold_home
        for _ in range(2):
            k = int(self.np_random.choice(len(spots), p=w / w.sum()))
            e = np.array([spots[k]["x"], spots[k]["y"], spots[k]["z"]], np.float64)
            if home:  # real-duel geometry: the agent at a hold spot of the OTHER side (1100-1600 units across)
                P = np.array([(sp["x"], sp["y"], sp["z"]) for sp in spots if sp["side"] != spots[k]["side"]], np.float64)
            d = np.linalg.norm(P - e, axis=1)
            cand = np.nonzero((d >= max(200.0, self.arena_min_dist)) & (d <= self.arena_max_dist))[0]
            for i in self.np_random.permutation(cand)[:tries]:
                a = np.asarray(P[i], np.float64)
                eye = a + (0.0, 0.0, 17.0)
                if min(self._trace([(eye, e + (0.0, 0.0, 10.0)), (eye, e + (0.0, 0.0, 28.0))])) >= 0.999:
                    return a, e
        return None

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

    def _los_cmds(self):
        """Navigator only: chest trace into aux[7] (as _los_cmd) plus eye -> head (+28) into aux[6]."""
        c = self._los_cmd()
        p = c.split()
        if p[2:] == ["0"] * 6:
            return c + ";cs2ai_trace 6 0 0 0 0 0 0"
        return c + f";cs2ai_trace 6 {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {float(p[7]) + 18:.0f}"

    def _nav_ext(self, s, ext):
        """Navigator labels (training only, never observed; present in info only after enable_nav()), 14 values:
        [0] forward key (0 none / 1 fwd / 2 back)  [1] side key (0 none / 1 left / 2 right)   -- the navigator's keys
        [2] wanted yaw bin  [3] wanted yaw turn (deg)  [4] wanted pitch bin  -- its look while the enemy is NOT visible
        [5] look valid (navigating: both alive, enemy not in line of sight)
        [6] keys valid (both alive; while engaging the keys are "stop"/counter-strafe)
        [7] own x, [8] own y scaled to [-1, 1] over the nav graph  [9] sin, [10] cos of the view yaw  ("where am I")
        [11] where-am-I valid (agent alive)  [12] mode (-1 none, 0 navigate, 1 engage)  [13] enemy in line of sight
        [14] which end this round started from (real duels only, agent alive): +1 T end, -1 CT end, 0 = no label
        Computed from the state after this step = the decision for the NEXT action (with reaction_delay 0 the frame
        the agent sees is this same state)."""
        out = np.zeros(15, np.float32)
        m = self._me(s)
        foes = [p for k, p in enumerate(s.players) if k != s.playernum and p.connected and p.alive]
        los = bool(m.alive and foes and (s.aux[7] >= 0.999 or s.aux[6] >= 0.999))
        self._nav_los = los
        enemy = None
        if m.alive and foes:
            enemy = min((np.array(p.origin) for p in foes), key=lambda o: float(np.linalg.norm(o - np.array(m.origin))))
        aim = (ext[3], ext[4], ext[8], ext[9])
        t = s.time - getattr(self, "_ep_start", s.time)
        if m.alive:
            a = self.nav.act(np.array(m.origin), (s.viewangles[0], s.viewangles[1]), tuple(s.velocity), t, los, enemy, aim)
        else:
            a = np.array([len(YAW_TABLE) // 2, len(PITCH_TABLE) // 2, 0, 0, 0, 0, 0, 0, 0])
        self.nav_action = a
        lab = self.nav.label
        if m.alive and enemy is not None:
            out[0], out[1] = lab["fwd"], lab["side"]
            out[6] = 1.0
            if lab["mode"] == 0:
                out[2], out[3], out[4], out[5] = lab["yaw_bin"], lab["yaw_deg"], lab["pitch_bin"], 1.0
            elif lab["mode"] == 1 and self.nav_engage_look and ext[9] > 0:
                # nav_engage_look (peek_v9): while engaging, the look label is the aim teacher's turn toward the enemy
                # (privileged, like the aim teacher itself) at the navigator's weight
                out[2], out[3], out[4], out[5] = ext[3], ext[6], ext[4], 1.0
        if m.alive:
            out[7:9] = self.nav.g.norm_xy(m.origin)
            yaw = math.radians(s.viewangles[1])
            out[9], out[10], out[11] = math.sin(yaw), math.cos(yaw), 1.0
        out[12] = lab.get("mode", -1) if (m.alive and enemy is not None) else -1
        out[13] = float(los)
        if self.scn.reset_round and m.alive:  # the agent starts every real-duel round at its own team's end
            out[14] = 1.0 if self.scn.human_team == "T" else -1.0
        return out

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
        tgt, err = self._aim_error(s) if (self.aim_shaping or self.offtarget_cost or self.acquire_bonus or self.hold_cost) else (None, None)
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
        ext = self._aux_ext(s)
        d = self._info_base(s, me, ext)
        if self.nav is not None:
            d["nav_ext"] = self._nav_ext(s, ext)
        return d

    def _info_base(self, s, me, ext):
        return {
            "time": s.time - getattr(self, "_ep_start", s.time),
            "kills": getattr(self, "_kills", 0),
            "deaths": getattr(self, "_deaths", 0),
            "damage": getattr(self, "_damage", 0),
            "health": me.health,
            "alive": bool(me.alive),
            "spawn_dist": self._ep_dist,
            "team": self.scn.human_team,
            "enemy_model": self.enemy_model,
            "spawn_offset": self._ep_offset,
            "no_contest": bool(getattr(self, "_no_contest", False)),
            "leashed": bool(self._leash is not None),
            "arena_hold": bool(getattr(self, "_arena_hold", False)),
            "go_look": bool(getattr(self, "_go_look", False)),
            "pocket_vis": self._pocket_vis,
            "acquire_time": self._acquired_t,
            "aux": self._aux_queue[0][:3].copy() if getattr(self, "_aux_queue", None) else np.zeros(3, np.float32),
            "aux_ext": ext,
            "los": bool(getattr(self, "_step_los", False)),
            "camp": None if self._camp is None else dict(
                unfreezes=self._camp["unfreezes"], homes=self._camp["homes"],
                **(dict({k: self._camp[k] for k in ("vis_steps", "live_steps", "vis_runs", "idle_vis_max")},
                        peeks=self._peeker.stats["peeks"], jumps=self._peeker.stats["jumps"],
                        jumps_refused=self._peeker.stats["jumps_refused"], jumps_seen=self._camp.get("jumps_seen", 0),
                        still=round(self._peeker.stats["still_ticks"] / max(1, self._peeker.stats["ticks"]), 3),
                        max_adv=round(float(self._peeker.stats["max_adv"]), 1), kind=self._camp.get("kind", "peek"),
                        spot=self._peeker.stats.get("spot"), switch_s=round(self._switch_steps * self.tick * self.repeat, 2),
                        switch_n=self._switch_n)
                   if self._camp["mode"] == "peek" and self._peeker is not None else {}),
                **({} if self._camp["mode"] != "side" else dict(
                    line_freezes=self._camp["line_freezes"], max_adv=round(float(self._camp["max_adv"]), 1),
                    crossed=bool(self._camp["crossed"]), walked=round(float(self._camp["walked"]))))),
        }
