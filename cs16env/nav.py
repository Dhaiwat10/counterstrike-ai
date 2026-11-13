"""Navigation graph, A* planner and the scripted NAVIGATOR teacher (pure Python/numpy, no engine needed).

Graph file: data/arena/MAP_nav.npz from scripts/make_nav_graph.py (nodes = walk-tested standing spots, edges walked
both ways in the engine, a 32-unit lookup grid "nearest node reachable in a straight line", node-to-node visibility).

The Navigator turns the true game state into the full 9-part action every agent step:
  1. enemy in line of sight  -> ENGAGE: stop (counter-strafe until nearly stationary), yaw/pitch/fire from the existing
     aim teacher (CS16Env._aux_ext), fire only once the horizontal speed is below `fire_speed`;
  2. otherwise               -> NAVIGATE: A* to a destination, walk the route, look down the route (furthest route
     node visible from the current waypoint), pitch level; turns at up to 48 deg per step from the first step;
  3. no progress for 0.6 s   -> RECOVER: side-step for 0.4 s, then re-plan.

Privileged inputs (what a player could NOT know from the screen) are listed in Navigator.__doc__.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

YAW_BINS = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, 24, 36, 48], dtype=np.float32)
PITCH_BINS = np.array([0, 0.25, 1, 3, 8, 16], dtype=np.float32)
YAW_TABLE = np.concatenate([-YAW_BINS[:0:-1], YAW_BINS])
PITCH_TABLE = np.concatenate([-PITCH_BINS[:0:-1], PITCH_BINS])

# spawn-side centres per map (map knowledge a player has): the "enemy side" destination of goal_mode="side"
MAP_SIDES = {"aim_map": {"T": (40.0, -680.0, -431.0), "CT": (-60.0, -2030.0, -431.0)}}

# the map's own spawn points (info_player_deathmatch = T, info_player_start = CT, read from the .bsp entity lump) and the
# yaw the map gives them; used by the real-duel options CS16Env.duel_random_spawn / duel_bot_mode="camp"
MAP_SPAWNS = {"aim_map": {
    "T": dict(yaw=90.0, pts=[(232, -777, -431), (232, -713, -431), (232, -585, -431), (104, -585, -431), (104, -713, -431),
                             (104, -777, -431), (-24, -777, -431), (-24, -713, -431), (-24, -585, -431), (-152, -585, -423),
                             (-152, -649, -423), (-152, -713, -423), (-152, -777, -423), (232, -649, -431), (104, -649, -431),
                             (-24, -649, -423)],
              # cover line of the T side (added 2026-11-13 for the model-as-CT work): the front face of the T row of boxes.
              # aim_map is point-symmetric about (-100, -1348.5) (the 16 T spawns are the CT spawns turned by 180 deg), so the
              # CT line y = -1790 maps to y = -907; engine traces northward hit the T boxes' front faces at y -904..-916
              # (one box sticks out to -949, the mirror of the CT box at -1748). "toward" = -1: the CT side is at smaller y.
              line_y=-907.0, toward=-1.0),
    "CT": dict(yaw=270.0, pts=[(-176, -2048, -423), (-304, -2048, -431), (-432, -2048, -431), (-48, -1920, -423),
                               (-64, -1984, -423), (-48, -2048, -423), (-48, -2112, -423), (-176, -2112, -431),
                               (-176, -1984, -431), (-176, -1920, -431), (-304, -1920, -431), (-304, -1984, -431),
                               (-304, -2112, -431), (-432, -2112, -431), (-432, -1984, -431), (-432, -1920, -431)],
               # cover line of the CT side: the front face of its row of boxes (y ~ -1790; the boxes span y -1790..-1880).
               # duel_bot_mode="side" keeps the bot's origin at y <= line_y ("toward" = +1: the T side is at larger y)
               line_y=-1790.0, toward=1.0),
}}

# YaPB player models (classes) per team, in the order of `yb add <difficulty> <personality> <team> <model>` (model 1..4;
# team 1 = T, 2 = CT). Checked by eye on aim_map 2026-11-13 (scratch/sides/probe/*_add_sheet.png).
TEAM_MODELS = {"T": ("terror", "leet", "arctic", "guerilla"), "CT": ("urban", "gsg9", "sas", "gign")}


def other_team(team):
    return "CT" if team == "T" else "T"

# the 8 movement key combinations (forward key, side key) and the direction each moves in, relative to the view yaw
# (degrees, counter-clockwise = left; env: forward 1 = fwd / 2 = back, side 1 = left / 2 = right)
KEY_COMBOS = [((1, 0), 0.0), ((1, 1), 45.0), ((0, 1), 90.0), ((2, 1), 135.0),
              ((2, 0), 180.0), ((2, 2), -135.0), ((0, 2), -90.0), ((1, 2), -45.0)]


def wrap(a):
    """Angle difference into [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def best_keys(want_yaw, view_yaw):
    """Which of the 8 key combinations moves closest to world direction `want_yaw` with the view at `view_yaw`.
    Returns ((fwd, side), angular error in degrees)."""
    rel = wrap(want_yaw - view_yaw)
    keys, ang = min(KEY_COMBOS, key=lambda kc: abs(wrap(rel - kc[1])))
    return keys, abs(wrap(rel - ang))


def yaw_bin(dyaw):
    return int(np.argmin(np.abs(YAW_TABLE - dyaw)))


def pitch_bin(dpitch_view):
    """Bin for a wanted change of the view pitch (the env turns the view pitch by -PITCH_TABLE[bin])."""
    return int(np.argmin(np.abs(-PITCH_TABLE - dpitch_view)))


def components(n, edges):
    """Connected-component label per node (union-find)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    roots = [find(i) for i in range(n)]
    _, lab = np.unique(roots, return_inverse=True)
    return lab


class NavGraph:
    def __init__(self, nodes, edges, grid=None, grid_origin=(0.0, 0.0), cell=32.0, vis=None):
        self.nodes = np.asarray(nodes, np.float64)
        self.n = len(self.nodes)
        self.edges = np.asarray(edges, np.int64).reshape(-1, 2)
        self.adj = [[] for _ in range(self.n)]
        for i, j in self.edges:
            w = float(np.linalg.norm(self.nodes[i] - self.nodes[j]))
            self.adj[i].append((int(j), w))
            self.adj[j].append((int(i), w))
        self.grid = None if grid is None else np.asarray(grid)
        self.origin = np.asarray(grid_origin, np.float64)
        self.cell = float(cell)
        self.vis = np.eye(self.n, dtype=bool) if vis is None else np.asarray(vis, bool)
        xy = self.nodes[:, :2]
        self.lo, self.hi = xy.min(0), xy.max(0)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["nodes"], z["edges"], z["grid"], z["grid_origin"], float(z["cell"]), z["vis"])

    def nearest(self, p):
        """Node to walk to first from point p: the lookup grid's straight-line-reachable node, else the nearest one."""
        p = np.asarray(p, np.float64)
        if self.grid is not None:
            gx, gy = ((p[:2] - self.origin) // self.cell).astype(int)
            if 0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
                # this cell, then its 8 neighbours (players stand near walls, where a cell centre can be off-floor)
                for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    y, x = gy + dy, gx + dx
                    if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1] and self.grid[y, x] >= 0:
                        return int(self.grid[y, x])
        return int(np.argmin(np.linalg.norm(self.nodes[:, :2] - p[:2], axis=1)))

    def astar(self, a, b):
        """Shortest node path a -> b (list of node ids, both ends included), or [] if unreachable."""
        if a == b:
            return [a]
        h = lambda k: float(np.linalg.norm(self.nodes[k] - self.nodes[b]))
        g = {a: 0.0}
        came = {}
        pq = [(h(a), 0.0, a)]
        done = set()
        while pq:
            _, gk, k = heapq.heappop(pq)
            if k in done:
                continue
            if k == b:
                path = [k]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return path[::-1]
            done.add(k)
            for j, w in self.adj[k]:
                ng = gk + w
                if ng < g.get(j, math.inf):
                    g[j], came[j] = ng, k
                    heapq.heappush(pq, (ng + h(j), ng, j))
        return []

    def path_len(self, path):
        return float(sum(np.linalg.norm(self.nodes[path[k + 1]] - self.nodes[path[k]]) for k in range(len(path) - 1)))

    def norm_xy(self, p):
        """Own position scaled to [-1, 1] over the graph's extent (the 'where am I' label)."""
        return 2.0 * (np.asarray(p[:2], np.float64) - self.lo) / np.maximum(self.hi - self.lo, 1.0) - 1.0


class Navigator:
    """Scripted orient-and-walk teacher. Call reset() at the start of a round and act() every agent step.

    Inputs of act():
      pos, vel        own origin and velocity                    -- player-knowable (map knowledge, own movement)
      view            own (pitch, yaw)                           -- player-knowable
      t               seconds since the round started            -- player-knowable
      los             enemy in line of sight (world trace)       -- ~knowable: it is on screen (or right beside it)
      enemy_pos       enemy origin (None if dead)                -- PRIVILEGED when los is False
      aim             (yaw bin, pitch bin, fire, valid) from the existing aim teacher -- privileged like that teacher
    Where hidden knowledge enters:
      goal_mode="enemy": the DESTINATION is the enemy's true position (privileged), re-planned every 0.5 s.
      goal_mode="side":  the destination is the last place the enemy was seen (if within `memory` s), else the enemy
                         spawn side (`enemy_side`) -- nothing hidden is used at all while navigating.
      In both modes the VIEW while navigating depends only on the route (map geometry + own position), never directly on
      the enemy's hidden position; in "enemy" mode the route itself is chosen with it.
      goal_mode="home" (HOME TEACHER, 2026-11-13): at the round start a spot of the hold table on the agent's OWN side is
                         drawn (set_home; weighted by how long the owner held there); it walks there by keys (same
                         route / look / recovery logic), then HOLDS: stops (counter-strafe), no movement keys, looks toward
                         where the enemy was last seen (within `memory` s) or else the enemy side, pitch level. It never
                         walks toward the enemy; after a fight it goes back to its spot. Nothing hidden is used.
      goal_mode="point": the same walk-then-hold to self.point, set from outside after reset() (test scripts).
    """

    KEYS_NONE = (0, 0)

    def __init__(self, graph: NavGraph, enemy_side=None, goal_mode="side", fire_speed=100.0, step_s=0.04,
                 memory=4.0, reach=48.0, look_max=1400.0, engage_max=1e9, look_cap=0.0, look_deadband=0.0,
                 look_follow=0.0):
        self.g = graph
        self.enemy_side = None if enemy_side is None else np.asarray(enemy_side, np.float64)
        self.goal_mode = goal_mode
        self.fire_speed = fire_speed
        self.dt = step_s
        self.memory = memory
        self.reach = reach
        self.look_max = look_max
        self.engage_max = engage_max
        # while NAVIGATING only (never while engaging): cap the turn per step at look_cap degrees (0 = off; 12 deg per
        # 0.04 s step = 300 deg/s, a fast human sweep) and treat wanted turns under look_deadband degrees as 0 (no tiny
        # alternating corrections when the look point shifts). Added 2026-11-12 after the model copied a "spin bot" search.
        self.look_cap = float(look_cap)
        self.look_deadband = float(look_deadband)
        # look_follow (deg, 0 = off): if the look point is more than this far from the walking direction, look where it
        # walks instead (no walking backwards while staring ahead past an obstacle; found 2026-11-13 at the T-side box)
        self.look_follow = float(look_follow)
        # goal_mode "home" / "point": the spot to walk to and hold, how close counts as there, where to look while holding
        self.point = None
        self.home_look = None
        self.hold_reach = 20.0
        self.hold_leave = 64.0
        # goal_mode "home": after this many seconds (uniform range) holding without seeing the enemy, move to another
        # own-side hold spot (a player repositions when nothing shows up; peek_v4 time-outs from the CT end). None = never.
        self.home_repick = (6.0, 10.0)
        self.reset()

    def set_home(self, spots, weights, look, rng, watch=None, watch_w=None, watch_lists=None):
        """goal_mode "home": draw the round's own-side hold spot (weighted) and the point to look at while holding.
        While holding with no recent sighting the teacher SCANS the other side's hold spots (map knowledge: where players
        hold): it looks at one (weighted) for 0.8-2.0 s, then another -- instead of staring at the enemy side's centre,
        which left enemies on the side spots (stone steps) ~50-80 deg off screen in peek_v1/v3's exams.
        watch_lists[k] = (points, weights) of the other side's spots that home spot k REALLY sees (engine traces,
        scripts/hold_spot_vis.py; preferred, since peek_v3). Fallback: watch / watch_w, kept if the nav graph says the
        spot is visible from home (approximate: it misses the raised steps)."""
        spots = np.asarray(spots, np.float64).reshape(-1, 3)
        w = np.asarray(weights, np.float64)
        k = int(rng.choice(len(spots), p=w / w.sum()))
        self._home = (spots, w, look, watch, watch_w, watch_lists)
        self._home_k = k
        self._hold_since = None
        self._holding = False
        self.point = spots[k].copy()
        self.home_look = None if look is None else np.asarray(look, np.float64)
        self.watch, self.watch_w, self._watch_k, self._watch_until, self._rng = None, None, None, -1.0, rng
        self._repick_s = float(rng.uniform(*self.home_repick)) if self.home_repick else None
        if watch_lists is not None:
            W, ww = watch_lists[k]
            if len(W):
                self.watch, self.watch_w = np.asarray(W, np.float64).reshape(-1, 3), np.asarray(ww, np.float64)
            return
        if watch is not None and len(watch):
            W = np.asarray(watch, np.float64).reshape(-1, 3)
            ww = np.asarray(watch_w if watch_w is not None else np.ones(len(W)), np.float64)
            here = self.g.nearest(self.point)
            vis = np.array([bool(self.g.vis[here, self.g.nearest(q)]) for q in W])
            if vis.any():
                self.watch, self.watch_w = W[vis], ww[vis]

    def _hold_look_point(self, t):
        """Where the holding teacher looks when it has not seen the enemy recently (see set_home)."""
        if getattr(self, "watch", None) is None:
            return self.home_look if self.home_look is not None else self.enemy_side
        if self._watch_k is None or t >= self._watch_until:
            p = self.watch_w / self.watch_w.sum()
            self._watch_k = int(self._rng.choice(len(self.watch), p=p))
            self._watch_until = t + float(self._rng.uniform(0.8, 2.0))
        return self.watch[self._watch_k]

    def reset(self):
        self.path, self.k, self.goal_node = [], 0, None
        self.t_plan = -1e9
        self.last_seen = None          # (position, time)
        self.hist = []                 # (t, pos) for the no-progress check
        self.recover_until, self.recover_dir = -1.0, 0.0
        self.stuck_events = 0
        self.stuck_log = []            # (t, position, waypoint node) of each no-progress event, for debugging
        self.no_progress_s = 0.0       # longest stretch without progress toward the goal
        self._best_goal_d, self._t_best = math.inf, 0.0
        self.label = {}
        self._following = False
        self._hold_since = None
        self._holding = False

    # ---------------------------------------------------------------- helpers
    def _goal_point(self, enemy_pos, t):
        if self.goal_mode in ("home", "point"):
            return self.point
        if self.goal_mode == "enemy" and enemy_pos is not None:
            return np.asarray(enemy_pos, np.float64)
        if self.last_seen is not None and t - self.last_seen[1] <= self.memory:
            return self.last_seen[0]
        return self.enemy_side

    def _plan(self, pos, goal, t, force=False):
        b = self.g.nearest(goal)
        if not force and b == self.goal_node and self.path:
            return
        # re-plans start from the waypoint we are already walking to (the grid's nearest node can lie behind us);
        # the first plan of a round and the plan after a recovery start from the grid lookup
        a = self.path[self.k] if (self.path and not force) else self.g.nearest(pos)
        self.goal_node = b
        self.path = self.g.astar(a, b)
        self.t_plan = t
        self.k = 0
        # skip a first node that lies behind us (the second one is closer)
        if len(self.path) > 1:
            n0, n1 = self.g.nodes[self.path[0]], self.g.nodes[self.path[1]]
            if np.linalg.norm(n1[:2] - pos[:2]) < np.linalg.norm(n1[:2] - n0[:2]):
                self.k = 1

    def _look_point(self, pos):
        """Furthest route node (within look_max of route length) visible from the current waypoint node and from the
        node nearest the player (a proxy for 'visible from here' without engine traces)."""
        if not self.path:
            return None
        k0 = min(self.k, len(self.path) - 1)
        w = self.path[k0]
        here = self.g.nearest(pos)
        best, L, prev = self.g.nodes[w], 0.0, self.g.nodes[w]
        for m in range(k0 + 1, len(self.path)):
            q = self.g.nodes[self.path[m]]
            L += float(np.linalg.norm(q - prev))
            prev = q
            if L > self.look_max:
                break
            if self.g.vis[w, self.path[m]] and self.g.vis[here, self.path[m]]:
                best = q
        if np.linalg.norm(best[:2] - pos[:2]) < 64.0 and self.goal_node is not None:
            best = self.g.nodes[self.goal_node]
        return best

    def _turn(self, view, look):
        """(yaw bin, pitch bin, wanted yaw change, wanted pitch change) to face `look` (world yaw) with a level view."""
        dy = wrap(look - view[1])
        if self.look_deadband > 0 and abs(dy) < self.look_deadband:
            dy = 0.0
        if self.look_cap > 0:
            dy = float(np.clip(dy, -self.look_cap, self.look_cap))
        dp = 0.0 - view[0]
        return yaw_bin(dy), pitch_bin(dp), dy, dp

    # ---------------------------------------------------------------- main
    def act(self, pos, view, vel, t, los, enemy_pos, aim=None):
        pos = np.asarray(pos, np.float64)
        speed = float(math.hypot(vel[0], vel[1]))
        a = np.zeros(9, np.int64)
        a[0], a[1] = len(YAW_TABLE) // 2, len(PITCH_TABLE) // 2
        lab = dict(mode=-1, fwd=0, side=0, yaw_deg=0.0, pitch_deg=0.0, yaw_bin=int(a[0]), pitch_bin=int(a[1]), look_valid=0)
        if enemy_pos is None:
            self.label = lab
            return a
        enemy_pos = np.asarray(enemy_pos, np.float64)
        dist = float(np.linalg.norm(enemy_pos - pos))
        if los:
            self.last_seen = (enemy_pos.copy(), t)
        if los and dist <= self.engage_max:
            # ENGAGE: counter-strafe to a stop, the aim teacher turns and fires
            keys = self.KEYS_NONE
            if speed > 40.0:
                keys, _ = best_keys(math.degrees(math.atan2(-vel[1], -vel[0])), view[1])
            if aim is not None and aim[3] > 0:
                a[0], a[1] = int(aim[0]), int(aim[1])
                a[4] = int(aim[2] > 0 and speed < self.fire_speed)
            a[2], a[3] = keys
            self.hist.clear()
            self._best_goal_d, self._t_best = math.inf, t
            lab.update(mode=1, fwd=keys[0], side=keys[1], yaw_bin=int(a[0]), pitch_bin=int(a[1]))
            self.label = lab
            return a

        # NAVIGATE
        goal = self._goal_point(enemy_pos, t)
        if goal is None:
            goal = enemy_pos
        if (self.goal_mode == "home" and getattr(self, "_repick_s", None) and self._hold_since is not None
                and t - max(self._hold_since, self.last_seen[1] if self.last_seen is not None else -1e9) > self._repick_s):
            spots, w, look, watch, watch_w, lists = self._home
            w2 = w.copy()
            if len(w2) > 1:
                w2[self._home_k] = 0.0
            self.set_home(spots, w2, look, self._rng, watch, watch_w, lists)
            self._home = (spots, w, look, watch, watch_w, lists)  # keep the original weights for the next move
            goal = self.point
            self.repicks = getattr(self, "repicks", 0) + 1
        gdist = float(np.linalg.norm(goal[:2] - pos[:2])) if self.goal_mode in ("home", "point") else 1e9
        # hysteresis (peek_v8): HOLD starts within hold_reach of the spot and lasts until it is pushed more than hold_leave
        # away. Without it a model jittering ~20 units around its spot got step-to-step flipping walk-back keys from the
        # teacher (keys agreement 0.18), never reached HOLD, and so never got the reposition timer either (greedy time-outs)
        self._holding = gdist < self.hold_reach or (getattr(self, "_holding", False) and gdist < self.hold_leave)
        if self._holding:
            if self._hold_since is None:
                self._hold_since = t
            # HOLD at the spot: stop (counter-strafe), look toward where the enemy was last seen, else the enemy side
            keys = self.KEYS_NONE
            if speed > 40.0:
                keys, _ = best_keys(math.degrees(math.atan2(-vel[1], -vel[0])), view[1])
            if self.last_seen is not None and t - self.last_seen[1] <= self.memory:
                lp = self.last_seen[0]
            else:
                lp = self._hold_look_point(t)
            look_yaw = view[1] if lp is None else math.degrees(math.atan2(lp[1] - pos[1], lp[0] - pos[0]))
            a[0], a[1], dy, dp = self._turn(view, look_yaw)
            a[2], a[3] = keys
            self.hist.clear()
            self._best_goal_d, self._t_best = math.inf, t
            lab.update(mode=0, fwd=keys[0], side=keys[1], yaw_deg=float(dy), pitch_deg=float(dp), yaw_bin=int(a[0]),
                       pitch_bin=int(a[1]), look_valid=1, hold=1)
            self.label = lab
            return a
        self._plan(pos, goal, t)
        # waypoint advance
        while self.k < len(self.path) - 1 and np.linalg.norm(self.g.nodes[self.path[self.k]][:2] - pos[:2]) < self.reach:
            self.k += 1
        wp = self.g.nodes[self.path[self.k]] if self.path else goal
        if self.path and self.k == len(self.path) - 1 and np.linalg.norm(wp[:2] - pos[:2]) < self.reach:
            wp = goal  # last node reached: head straight for the destination point
        move_yaw = math.degrees(math.atan2(wp[1] - pos[1], wp[0] - pos[0]))

        # progress / stuck bookkeeping
        gd = self.g.path_len(self.path[self.k:]) + float(np.linalg.norm(wp[:2] - pos[:2])) if self.path else 0.0
        if gd < self._best_goal_d - 8.0:
            self._best_goal_d, self._t_best = gd, t
        self.no_progress_s = max(self.no_progress_s, t - self._t_best) if gd > 2 * self.reach else self.no_progress_s
        self.hist.append((t, pos.copy()))
        while self.hist and t - self.hist[0][0] > 0.6:
            self.hist.pop(0)
        if t >= self.recover_until and len(self.hist) > 1 and t - self.hist[0][0] >= 0.56 and gd > self.reach:
            if np.linalg.norm(self.hist[-1][1][:2] - self.hist[0][1][:2]) < 20.0:
                self.stuck_events += 1
                self.stuck_log.append((round(t, 2), [round(float(x)) for x in pos], int(self.path[self.k]) if self.path else -1))
                self.recover_until = t + 0.4
                self.recover_dir = move_yaw + (90.0 if self.stuck_events % 2 else -90.0)
                self.hist.clear()
                self.t_plan = -1e9  # re-plan once recovered
        if t < self.recover_until:
            move_yaw = self.recover_dir
        elif self.t_plan < -1e8:
            self._plan(pos, goal, t, force=True)

        look = self._look_point(pos)
        look_yaw = move_yaw if look is None else math.degrees(math.atan2(look[1] - pos[1], look[0] - pos[0]))
        home_look = self.goal_mode == "home" and getattr(self, "_rng", None) is not None
        if home_look:
            # home teacher (peek_v5): while walking to / between its own-side spots it faces where enemies can see it
            # from (the scanned hold spots, or the last sighting) and moves with whatever keys fit -- in peek_v4 it walked
            # with its back to the other side and was shot from ~80 deg off screen
            lp = self.last_seen[0] if (self.last_seen is not None and t - self.last_seen[1] <= self.memory) else self._hold_look_point(t)
            if lp is not None:
                look_yaw = math.degrees(math.atan2(lp[1] - pos[1], lp[0] - pos[0]))
        if self.look_follow > 0 and not home_look:
            # with hysteresis: start looking where it walks beyond look_follow deg, go back to the route view only once the
            # two are within look_follow - 40 deg (no step-to-step switching near the threshold)
            off = abs(wrap(look_yaw - move_yaw))
            if self._following and off < self.look_follow - 40.0:
                self._following = False
            elif not self._following and off > self.look_follow:
                self._following = True
            if self._following:
                look_yaw = move_yaw
        a[0], a[1], dy, dp = self._turn(view, look_yaw)
        keys, _ = best_keys(move_yaw, view[1] + YAW_TABLE[a[0]])  # keys for the view after this step's turn
        a[2], a[3] = keys
        lab.update(mode=0, fwd=keys[0], side=keys[1], yaw_deg=float(dy), pitch_deg=float(dp), yaw_bin=int(a[0]),
                   pitch_bin=int(a[1]), look_valid=1)
        self.label = lab
        return a
