"""Training scenarios: server/bot settings + episode rules.

Every scenario runs on a listen server inside the agent's own engine process, with YaPB
bots as opponents. Server-side state is only used for rewards / episode bookkeeping; the
agent observes pixels only.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ReGameDLL settings shared by all deathmatch-style scenarios: endless round, instant
# respawn, no buying/economy, no objectives, no freeze time.
_DM_BASE = [
    "mp_round_infinite 1",
    "mp_forcerespawn 1",
    "mp_respawn_immunitytime 0",
    "mp_freezetime 0",
    "mp_buytime 0",
    "mp_give_player_c4 0",
    "mp_infinite_ammo 2",
    "mp_timelimit 0",
    "mp_maxrounds 0",
    "mp_limitteams 0",
    "mp_autoteambalance 0",
    "mp_auto_join_team 1",
    "mp_kill_filled_spawn 1",
    "mp_item_staytime 0",
    "sv_alltalk 0",
    "yb_display_welcome_text 0",
    "yb_chat 0",
    "yb_radio_mode 0",
    "yb_ignore_objectives 1",
    "yb_economics_rounds 0",
    "yb_quota_mode normal",
    "hud_takesshots 0",
]


@dataclass
class Scenario:
    name: str
    map: str
    server_cmds: list[str]
    bots: int
    bot_difficulty: int            # YaPB 0 (newbie) .. 4 (expert)
    episode_seconds: float         # game-time length of an episode
    end_on_death: bool = False
    end_on_kill: bool = False
    human_team: str = "T"
    reset_round: bool = False      # restart the round (respawn everyone at spawn points) on every reset
    arena: str = ""                # aim arena: npz of line-of-sight spawn pairs (scripts/make_arena_pairs.py)
    arena_max_dist: float = 2000.0 # curriculum: max spawn distance (units)
    arena_max_offset: float = 30.0 # curriculum: max initial aim error (degrees, yaw; pitch gets 1/4)
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def setup_commands(self) -> list[str]:
        return [
            *self.server_cmds,
            f"humans_join_team {self.human_team}",
            f"yb_difficulty {self.bot_difficulty}",
            f"yb_quota {self.bots}",
        ]


SCENARIOS: dict[str, Scenario] = {
    # Free-for-all deathmatch on Dust2: every bot is an enemy, AK + deagle for everyone.
    "dm_dust2": Scenario(
        name="dm_dust2", map="de_dust2", bots=8, bot_difficulty=1, episode_seconds=120.0,
        server_cmds=_DM_BASE + [
            "mp_freeforall 1",
            "mp_t_default_weapons_primary ak47", "mp_ct_default_weapons_primary ak47",
            "mp_t_default_weapons_secondary deagle", "mp_ct_default_weapons_secondary deagle",
        ],
        notes="FFA deathmatch, fixed-length episodes, instant respawn.",
    ),
    # One bot, one life each: the closest stock-map stand-in for an aim duel.
    "duel_dust2": Scenario(
        name="duel_dust2", map="de_dust2", bots=1, bot_difficulty=1, episode_seconds=60.0,
        end_on_death=True, end_on_kill=True, reset_round=True,
        server_cmds=_DM_BASE + [
            "mp_freeforall 1",
            "mp_t_default_weapons_primary ak47", "mp_ct_default_weapons_primary ak47",
        ],
        notes="1v1 vs a single bot; episode ends at the first kill or death (or timeout).",
    ),
    # Aim arena: one bot placed in view at a random line-of-sight spot; episode = one engagement.
    "aim_dust2": Scenario(
        name="aim_dust2", map="de_dust2", bots=1, bot_difficulty=0, episode_seconds=4.0,
        end_on_death=True, end_on_kill=True, arena="data/arena/dust2_pairs.npz",
        server_cmds=_DM_BASE + [
            "mp_freeforall 1",
            "mp_t_default_weapons_primary ak47", "mp_ct_default_weapons_primary ak47",
            "mp_t_default_weapons_secondary \"\"", "mp_ct_default_weapons_secondary \"\"",
            # magazine never empties: teleporting between rounds doesn't refill it, so a surviving agent used to drain
            # it over ~8-10 rounds and start fights in a forced 2.5 s reload. One round rarely uses >30 bullets, so
            # this is ~"full magazine every round". Revisit once the reload head is learned.
            "mp_infinite_ammo 1",
            # NOTE: keep the shared `mp_forcerespawn 1`. The 1 s of game time between a kill/death and the next round
            # (~100 engine ticks, 107 ms of a restart) looks like dead time, but it is the cool-down that makes
            # rounds independent: with 0.05 the scripted delayed aimer fell from 97% to 58% kills on the 120 ms exam
            # (deaths 3% -> 39%, no-contests 5 -> 27 per 150 rounds; 2026-11-12), i.e. the survivor carries its
            # combat state (recoil, the bot's lock on the agent) into the next round.
        ],
        notes="Phase 3 stage 1: enemy in view, ends on kill/death/4 s. Distance and initial aim error are curriculum knobs.",
    ),
    # Navigation / exploration warm-up: no enemies.
    "explore_dust2": Scenario(
        name="explore_dust2", map="de_dust2", bots=0, bot_difficulty=0, episode_seconds=60.0,
        server_cmds=_DM_BASE,
        notes="Empty map, useful for testing movement and the env itself.",
    ),
}

# The same aim arena on community aim maps (added 2026-11-12; maps from GameBanana, bot navigation built locally by
# YaPB's analyser, spawn pairs by scripts/make_arena_pairs.py MAP OUT STEPS). Both players are handed a rifle: the
# rifles on the ground need walking, which the agent cannot do yet.
for _name, _map, _pairs in (("aim_aimmap", "aim_map", "aim_map_pairs.npz"), ("aim_akcolt", "aim_ak-colt", "aim_ak_colt_pairs.npz")):
    _base = SCENARIOS["aim_dust2"]
    SCENARIOS[_name] = Scenario(
        name=_name, map=_map, bots=_base.bots, bot_difficulty=_base.bot_difficulty, episode_seconds=_base.episode_seconds,
        end_on_death=True, end_on_kill=True, arena=f"data/arena/{_pairs}", server_cmds=list(_base.server_cmds),
        notes=f"Aim arena on {_map}: enemy placed in view, ends on kill/death/4 s.")
