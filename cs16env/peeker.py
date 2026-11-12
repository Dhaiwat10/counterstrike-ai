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


class Peeker:
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
        self.react, self.rate, self.err = AIM.get(int(difficulty), AIM[0])
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
        # ---- view
        eye = self.pos + (0.0, 0.0, 17.0)
        if los:
            if self.seen_t is None:
                self.seen_t = self.t + float(self.rng.uniform(*self.react))
                e = self.err
                self.aim_off = (float(self.rng.uniform(-e, e)), float(self.rng.uniform(-e, e)) / 2)
            if self.t >= self.seen_t:
                d = np.asarray(player_chest, np.float64) - eye
                ty = math.degrees(math.atan2(d[1], d[0])) + self.aim_off[0]
                tp = -math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))) + self.aim_off[1]
                self._turn(ty, tp, self.rate * dt)
        else:
            self.seen_t = None
            if self.look_point is not None:
                d = self.look_point - eye
                self._turn(math.degrees(math.atan2(d[1], d[0])), 0.0, 0.5 * self.rate * dt)
        return self.pos.copy(), self.pitch, self.yaw, jumped

    def _turn(self, ty, tp, cap):
        dy = wrap(ty - self.yaw)
        dp = tp - self.pitch
        n = math.hypot(dy, dp)
        if n > cap:
            dy, dp = dy * cap / n, dp * cap / n
        self.yaw = (self.yaw + dy) % 360.0
        self.pitch += dp
