"""Scripted PEEKER body for the real-duel opponent (CS16Env.duel_bot_mode = "peek"; pure Python, no engine).

Why scripted: the owner wants an opponent that stays behind its own cover line and peeks: comes out from behind its boxes,
takes the fight, goes back, from varied spots and with varied timing. YaPB alone walks over to the player, and our engine
cannot read a bot's view angles, so any position control also sets its view. So here we move the bot's BODY and VIEW
ourselves (one engine command per tick), and YaPB (not frozen) still decides by itself WHEN to fire.

Body: cycles  wait behind cover -> walk out to a peek spot -> hold there (0.4-2.5 s) -> walk back -> wait (0.5-3 s) -> ...
  Spots come from the navigation graph, only from the bot's own area (behind its cover line by >= `margin` units):
  a COVER spot cannot be seen from the player's nearest graph spot, a PEEK spot is linked to it by a walk-tested edge and
  can be seen. Pairs, speeds (150-230 u/s) and times are random each cycle (no fixed loop to memorise). Long moves between
  cover spots happen by a jump only while the player cannot see the bot and cannot see the destination; otherwise it walks.
View: while it sees the player (line of sight): after a reaction time it turns toward the player's chest at a limited turn
  rate and settles with a small fixed error (both by difficulty); otherwise it looks toward the other side.
Privileged inputs: the player's position (to choose cover / peek spots and to aim, like YaPB's own aim) -- the opponent may
use game state; the trained model never sees any of it.
"""
from __future__ import annotations

import math

import numpy as np

from .nav import components, wrap

# by YaPB difficulty: reaction time range (s), max turn rate (deg/s), settled aim error (deg, yaw; pitch gets half)
AIM = {0: ((0.15, 0.35), 360.0, 1.5), 1: ((0.12, 0.28), 500.0, 1.1), 2: ((0.10, 0.22), 700.0, 0.8),
       3: ((0.08, 0.18), 900.0, 0.6), 4: ((0.06, 0.14), 1100.0, 0.4)}
# "THE SWITCH" (owner design, 2026-11-13; CS16Env.duel_open_switch): while the player is past its own cover line the scripted
# opponent aims with these settings and PRE-AIMS (turns toward the player's chest even through walls, no reaction delay),
# so it is already on target when line of sight appears. Values from the open-deadliness measurement (scratch/peek_log.md).
DEADLY_AIM = ((0.0, 0.0), 2000.0, 0.3)
# deadly mode's scripted trigger (YaPB's own trigger hesitates for seconds even with perfect aim, scratch/peek_log.md):
# fire when the view is within FIRE_TOL deg of its aim point, bursts of FIRE_BURST s, FIRE_PAUSE s apart
# taps: one bullet per press (first-shot accuracy), aimed at the HEAD (chest point + DEADLY_HEAD_DZ units; one rifle
# hit there kills) -- found 2026-11-13: 0.3 s bursts at the chest landed ~1 hit in 7 at 800-1300 units
FIRE_TOL, FIRE_BURST, FIRE_PAUSE = 1.0, 0.02, 0.14   # pause 0.08 / 0.14 / 0.2 / 0.28 tested: 0.14 keeps ~13 HP per tap
DEADLY_MOVE_AFTER = 0.6  # hold-spot opponent with the switch on: changes (hidden) spot after 0.6 s without sight, not 2-6 s
DEADLY_HEAD_DZ = 8.0   # 15 / 8 / 0 / -4 tested: 8 best with the body in the air; see scratch/peek_log.md


class _Aimer:
    """View of a scripted opponent body: reaction delay, limited turn rate, small fixed error (AIM row), optional
    deadly mode (DEADLY_AIM + pre-aim). Subclasses set self.pos / self.yaw / self.pitch / self.t / self.look_point."""

    def _aim_init(self, difficulty):
        self.base_aim = AIM.get(int(difficulty), AIM[0])
        self.react, self.rate, self.err = self.base_aim
        self.deadly = False
        self.deadly_aim = DEADLY_AIM
        self.deadly_preaim = True
        self.on_target = False     # the view is within FIRE_TOL of where it wants to aim (set by _view)
        self.fire_t = 0.0          # trigger cadence clock (see trigger)
        self.head_dz = DEADLY_HEAD_DZ

    def set_deadly(self, on, aim=None, preaim=None):
        """Switch the deadly settings on/off (takes effect on the next tick; a new aim error is drawn)."""
        if aim is not None:
            self.deadly_aim = aim
        if preaim is not None:
            self.deadly_preaim = bool(preaim)
        on = bool(on)
        if on != self.deadly:
            self.deadly = on
            self.react, self.rate, self.err = self.deadly_aim if on else self.base_aim
            self.seen_t = None  # re-draw the reaction time and aim error under the new settings

    def _view(self, dt, player_chest, los):
        eye = self.pos + (0.0, 0.0, 17.0)
        if los or (self.deadly and self.deadly_preaim):
            if self.seen_t is None:
                self.seen_t = self.t + float(self.rng.uniform(*self.react))
                e = self.err
                self.aim_off = (float(self.rng.uniform(-e, e)), float(self.rng.uniform(-e, e)) / 2)
            self.on_target = False
            if self.t >= self.seen_t:
                d = np.asarray(player_chest, np.float64) + (0.0, 0.0, self.head_dz if self.deadly else 0.0) - eye
                ty = math.degrees(math.atan2(d[1], d[0])) + self.aim_off[0]
                tp = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))) + self.aim_off[1]
                self._turn(ty, tp, self.rate * dt)
                self.on_target = math.hypot(wrap(ty - self.yaw), tp - self.pitch) <= FIRE_TOL
        else:
            self.seen_t = None
            self.on_target = False
            if self.look_point is not None:
                d = self.look_point - eye
                self._turn(math.degrees(math.atan2(d[1], d[0])), 0.0, 0.5 * self.rate * dt)

    def trigger(self, dt, los):
        """Scripted trigger, deadly mode only (CS16Env.open_switch_fire; engine command cs2ai_fire): with line of sight and
        the view on target, press attack in bursts of FIRE_BURST s with FIRE_PAUSE s between (rifle accuracy recovers)."""
        if not (self.deadly and los and self.on_target):
            self.fire_t = 0.0
            return False
        self.fire_t += dt
        return (self.fire_t % (FIRE_BURST + FIRE_PAUSE)) < FIRE_BURST

    def _turn(self, ty, tp, cap):
        dy = wrap(ty - self.yaw)
        dp = tp - self.pitch
        n = math.hypot(dy, dp)
        if n > cap:
            dy, dp = dy * cap / n, dp * cap / n
        self.yaw = (self.yaw + dy) % 360.0
        self.pitch += dp


class Peeker(_Aimer):
    def __init__(self, graph, line_y, toward, rng, difficulty=0, margin=20.0, area_depth=420.0, look_point=None):
        self.g = graph
        self.rng = rng
        self.line_y, self.toward = float(line_y), float(toward)
        lab = components(graph.n, graph.edges)
        main = np.bincount(lab).argmax()
        adv = (graph.nodes[:, 1] - self.line_y) * self.toward
        # own area: behind the line by >= margin, at most area_depth deep, ground level of the main graph component
        self.area = np.nonzero((adv <= -margin) & (adv >= -area_depth) & (lab == main))[0]
        aset = set(int(k) for k in self.area)
        self.nbr = {int(k): [j for j, _ in graph.adj[k] if j in aset] for k in self.area}
        self.look_point = None if look_point is None else np.asarray(look_point, np.float64)
        self._aim_init(difficulty)
        self.margin = margin

    # ------------------------------------------------------------------ helpers
    def _pairs(self, here_node):
        vis = self.g.vis[here_node]
        # robust cover: hidden from the player's spot AND from the spots next to it (node visibility is approximate)
        near = [here_node] + [j for j, _ in self.g.adj[here_node]]
        hidden_any = ~self.g.vis[near].any(0)
        out = []
        for h in self.area:
            if not hidden_any[h]:
                continue
            for p in self.nbr[int(h)]:
                if vis[p]:
                    out.append((int(h), int(p)))
        return out

    def _route(self, a, b):
        """Shortest route inside the own area from node a to b (Dijkstra over area edges only); [] if none."""
        import heapq
        if a not in self.nbr or b not in self.nbr:
            return []
        dist, came, pq = {a: 0.0}, {}, [(0.0, a)]
        while pq:
            dk, k = heapq.heappop(pq)
            if k == b:
                path = [b]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return path[::-1]
            if dk > dist.get(k, math.inf):
                continue
            for j in self.nbr[k]:
                nd = dk + float(np.linalg.norm(self.g.nodes[j] - self.g.nodes[k]))
                if nd < dist.get(j, math.inf):
                    dist[j], came[j] = nd, k
                    heapq.heappush(pq, (nd, j))
        return []

    def reset(self, player_pos):
        here = self.g.nearest(player_pos)
        hidden = [int(k) for k in self.area if not self.g.vis[here, k]] or [int(k) for k in self.area]
        self.node = hidden[int(self.rng.integers(len(hidden)))]
        self.pos = self.g.nodes[self.node].copy()
        self.yaw, self.pitch = 0.0, 0.0
        if self.look_point is not None:
            d = self.look_point - self.pos
            self.yaw = math.degrees(math.atan2(d[1], d[0])) + float(self.rng.uniform(-25, 25))
        self.phase, self.t_phase = "wait", float(self.rng.uniform(0.3, 2.0))
        self.legs = []          # remaining points to walk through
        self.speed = 200.0
        self.peek = None        # (cover node, peek node) of the current cycle
        self.seen_t = None      # time the current sighting started
        self.aim_off = (0.0, 0.0)
        self.t = 0.0
        self.stats = dict(peeks=0, jumps=0, jumps_refused=0, still_ticks=0, ticks=0, max_adv=-1e9)
        self.jump, self.jump_verdict, self.jump_traced = None, None, False
        self.unseen_t = 0.0
        self.last_los_t = -1e9
        return self.pos.copy()

    def _start_cycle(self, player_pos, los_now):
        here = self.g.nearest(player_pos)
        pairs = self._pairs(here)
        if not pairs:  # no cover/peek pair for where the player is: wander to a random area spot
            tgt = int(self.area[int(self.rng.integers(len(self.area)))])
            r = self._route(self.node, tgt)
            self.legs = [self.g.nodes[k].copy() for k in r[1:]]
            self.peek, self.phase = None, "walk"
            self.speed = float(self.rng.uniform(150, 230))
            return
        h, p = pairs[int(self.rng.integers(len(pairs)))]
        self.peek = (h, p)
        self.speed = float(self.rng.uniform(150, 230))
        r = self._route(self.node, h)
        far = sum(np.linalg.norm(self.g.nodes[r[k + 1]] - self.g.nodes[r[k]]) for k in range(len(r) - 1)) if r else 1e9
        near = [here] + [j for j, _ in self.g.adj[here]]
        if (far > 300 or not r) and self.unseen_t >= 0.3 and not self.g.vis[near, h].any():
            # a jump is wanted (unseen, to a spot the map table says is unseen); the env confirms it with live traces
            # from the player's eye to the bot's current and new spot (chest and head) before it happens: JUMPCHECK
            self.jump, self.jump_verdict, self.jump_traced = (h, p, r), None, False
            self.phase = "jumpcheck"
            return
        self.legs = [self.g.nodes[k].copy() for k in r[1:]] + [self.g.nodes[p].copy()]
        self.phase = "out"

    # ------------------------------------------------------------------ main
    def tick(self, dt, player_pos, player_chest, los):
        """Advance one engine tick. los = the player and the bot see each other (from the latest trace).
        Returns (position, pitch, yaw, jumped) for the bot."""
        self.t += dt
        self.unseen_t = 0.0 if los else self.unseen_t + dt
        jumped = False
        prev = self.pos.copy()
        if los:
            self.last_los_t = self.t
        if self.deadly:  # the switch is on: go and take the shot now; stop and hold where it is while the player is in view
            if self.phase in ("wait", "settle"):
                self.t_phase = min(self.t_phase, 0.0)
            elif self.phase in ("out", "back", "walk") and los:  # seen while walking: stop here (behind its line), shoot
                self.legs, self.phase, self.t_phase = [], "hold", 0.2
            if self.phase == "hold" and self.t - self.last_los_t < 1.0:  # keep the angle until 1 s after the last sight
                self.t_phase = max(self.t_phase, 0.2)
        if self.phase == "settle":
            self.t_phase -= dt
            if self.t_phase <= 0:
                self.phase = "out"
        if self.phase in ("wait", "hold"):
            self.t_phase -= dt
            if self.t_phase <= 0:
                if self.phase == "hold" and self.peek is not None:
                    self.legs = [self.g.nodes[self.peek[0]].copy()]
                    self.phase = "back"
                else:
                    self._start_cycle(player_pos, los)
        if self.phase == "jumpcheck" and self.jump_verdict is not None:
            h, p, r = self.jump
            if self.jump_verdict and self.unseen_t >= 0.3:  # both spots confirmed hidden from the player's eye: jump
                self.pos, self.node = self.g.nodes[h].copy(), h
                self.stats["jumps"] += 1
                jumped = True
                self.legs = [self.g.nodes[p].copy()]
                self.jump, self.jump_verdict = None, None
                self.phase, self.t_phase = "settle", float(self.rng.uniform(0.5, 1.5))  # pause in cover before peeking
            else:  # walk instead (or, without a route, wait and try another cycle)
                self.stats["jumps_refused"] += 1
                self.legs = [self.g.nodes[k].copy() for k in r[1:]] + [self.g.nodes[p].copy()] if r else []
            if self.phase == "jumpcheck":
                self.jump, self.jump_verdict = None, None
                self.phase = "out" if self.legs else "wait"
                self.t_phase = float(self.rng.uniform(0.5, 1.5))
        if self.phase in ("out", "back", "walk"):
            step = self.speed * dt
            while step > 0 and self.legs:
                d = self.legs[0] - self.pos
                L = float(np.linalg.norm(d[:2]))
                if L <= step:
                    self.pos = self.legs.pop(0)
                    step -= L
                else:
                    self.pos = self.pos + d * (step / max(L, 1e-6))
                    step = 0
            if not self.legs:
                self.node = self.g.nearest(self.pos)
                if self.phase == "out":
                    self.phase, self.t_phase = "hold", float(self.rng.uniform(0.3, 1.3))
                elif self.phase == "back":
                    self.stats["peeks"] += 1
                    self.phase, self.t_phase = "wait", float(self.rng.uniform(0.8, 3.0))
                else:
                    self.phase, self.t_phase = "wait", float(self.rng.uniform(0.8, 3.0))
        adv = (self.pos[1] - self.line_y) * self.toward
        assert adv <= 0.0, "peeker would cross its cover line"
        self.stats["max_adv"] = max(self.stats["max_adv"], adv)
        self.stats["ticks"] += 1
        self.stats["still_ticks"] += int(np.linalg.norm(self.pos[:2] - prev[:2]) < 1e-3)
        self._view(dt, player_chest, los)
        return self.pos.copy(), self.pitch, self.yaw, jumped

    def jump_target(self):
        """Where a requested jump (phase "jumpcheck") would put the body."""
        return self.g.nodes[self.jump[0]]


# ------------------------------------------------------------------------------------------------------------------ hold
def load_hold_spots(path=None):
    """The hold-spot table (scripts/make_hold_spots.py): list of dicts {x, y, z, w, side, raised, ...}."""
    import json
    from pathlib import Path
    path = path or Path(__file__).parent / "aim_map_hold_spots.json"
    with open(path) as fh:
        return json.load(fh)["spots"]


def side_spots(spots, side, line_y, toward, margin=0.0):
    """(N,3) positions and (N,) weights of the table's spots on `side` that are behind that side's line by >= margin."""
    P = np.array([(s["x"], s["y"], s["z"]) for s in spots if s["side"] == side], np.float64).reshape(-1, 3)
    W = np.array([s["w"] for s in spots if s["side"] == side], np.float64)
    keep = (P[:, 1] - line_y) * toward <= -margin
    return P[keep], W[keep]


class HoldBot(_Aimer):
    """Scripted HOLD-SPOT opponent body (CS16Env.duel_bot_mode = "hold"; pure Python, no engine).

    Stands at one spot of the hold table (where the owner stood while in line of sight, mirrored to both ends; raised
    spots such as the stone steps included) on its own side, behind its cover line. View: as the peeker (_Aimer): looks
    toward the other side; with line of sight, after a reaction time, turns toward the player's chest at a limited rate
    with a small error. YaPB (not frozen) decides by itself WHEN to fire. While the player has not seen it for a random
    2-6 s (0.6 s with the switch on) it may change spot: the env confirms with live traces from the player's eye that both the current and the new
    spot are hidden (phase "jumpcheck", like the peeker's jump); then the body moves there at once. Never past its line.
    """

    def __init__(self, spots, weights, line_y, toward, rng, difficulty=0, look_point=None, move_after=(2.0, 6.0)):
        self.P = np.asarray(spots, np.float64).reshape(-1, 3)
        self.W = np.asarray(weights, np.float64)
        assert len(self.P) > 0, "no hold spots on this side"
        self.line_y, self.toward = float(line_y), float(toward)
        assert np.all((self.P[:, 1] - self.line_y) * self.toward <= 0.0), "hold spot past the cover line"
        self.rng = rng
        self.look_point = None if look_point is None else np.asarray(look_point, np.float64)
        self.move_after = move_after
        self._aim_init(difficulty)

    def _draw(self, exclude=None):
        w = self.W.copy()
        if exclude is not None and len(w) > 1:
            w[exclude] = 0.0
        return int(self.rng.choice(len(w), p=w / w.sum()))

    def reset(self, player_pos=None):
        self.k = self._draw()
        self.pos = self.P[self.k].copy()
        self.yaw, self.pitch = 0.0, 0.0
        if self.look_point is not None:
            d = self.look_point - self.pos
            self.yaw = math.degrees(math.atan2(d[1], d[0])) + float(self.rng.uniform(-25, 25))
        self.phase = "hold"
        self.seen_t = None
        self.aim_off = (0.0, 0.0)
        self.t = 0.0
        self.unseen_t = 0.0
        self.next_move = float(self.rng.uniform(*self.move_after))
        self.jump, self.jump_verdict, self.jump_traced = None, None, False
        self.stats = dict(peeks=0, jumps=0, jumps_refused=0, still_ticks=0, ticks=0, max_adv=-1e9, spot=int(self.k))
        return self.pos.copy()

    def jump_target(self):
        return self.P[self.jump]

    def tick(self, dt, player_pos, player_chest, los):
        self.t += dt
        self.unseen_t = 0.0 if los else self.unseen_t + dt
        jumped = False
        wait = min(self.next_move, DEADLY_MOVE_AFTER) if self.deadly else self.next_move
        if self.phase == "hold" and len(self.P) > 1 and self.unseen_t >= wait:
            self.jump, self.jump_verdict, self.jump_traced = self._draw(exclude=self.k), None, False
            self.phase = "jumpcheck"
        elif self.phase == "jumpcheck" and self.jump_verdict is not None:
            if self.jump_verdict and self.unseen_t >= 0.3:
                self.k = self.jump
                self.pos = self.P[self.k].copy()
                self.stats["jumps"] += 1
                jumped = True
            else:
                self.stats["jumps_refused"] += 1
            self.jump, self.jump_verdict = None, None
            self.phase = "hold"
            self.unseen_t = 0.0
            self.next_move = float(self.rng.uniform(*self.move_after))
        adv = (self.pos[1] - self.line_y) * self.toward
        assert adv <= 0.0, "hold bot would cross its cover line"
        self.stats["max_adv"] = max(self.stats["max_adv"], adv)
        self.stats["ticks"] += 1
        self.stats["still_ticks"] += 1
        self._view(dt, player_chest, los)
        return self.pos.copy(), self.pitch, self.yaw, jumped
