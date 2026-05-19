"""
Orbit Wars Local Arena — Test agents before uploading to Kaggle.

Features:
  - Head-to-head matchups between any two agents
  - Round-robin tournaments
  - Win/loss/tie stats with ship-count margins
  - Game replays saved as JSON
  - Alternates starting positions for fairness
  - Accepts both .py and .ipynb files directly

Usage:
  python arena.py submission.py submission_958_raw.py --games 20
  python arena.py orbit-wars-agent-v11.ipynb orbit-wars-agent-958.1.ipynb --games 10
  python arena.py --tournament agent1.ipynb agent2.py agent3.ipynb --games 6
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Sequence
from kaggle_environments import make

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook_util import resolve_agent_path
from core.telemetry import reset as reset_telemetry, snapshot as telemetry_snapshot
from match_stats import aggregate_game_stats, summarize_game_stats


@dataclass(frozen=True, slots=True)
class AgentSpec:
    raw: str
    ref: Any
    name: str
    index: int


def _unique_names(names: Sequence[str]) -> list[str]:
    counts = defaultdict(int)
    unique = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            unique.append(name)
        else:
            unique.append(f"{name} ({counts[name]})")
    return unique


def resolve_agent_specs(agent_paths: Sequence[str]) -> list[AgentSpec]:
    resolved = []
    for index, raw in enumerate(agent_paths):
        ref, name = resolve_agent_path(raw)
        resolved.append(AgentSpec(raw=raw, ref=ref, name=name, index=index))
    unique_names = _unique_names([spec.name for spec in resolved])
    return [
        AgentSpec(raw=spec.raw, ref=spec.ref, name=unique_names[i], index=spec.index)
        for i, spec in enumerate(resolved)
    ]


def get_final_ship_counts(result):
    """Extract final ship counts from game result."""
    final = result[-1]
    counts = []
    for i, agent_result in enumerate(final):
        obs = agent_result.get("observation")
        if obs is None:
            counts.append(0)
            continue

        ships = 0
        player = i
        raw_planets = obs.get("planets", []) or []
        raw_fleets = obs.get("fleets", []) or []

        for p in raw_planets:
            if p[1] == player:  # owner == player
                ships += int(p[5])  # ships field
        for f in raw_fleets:
            if f[1] == player:  # owner == player
                ships += int(f[6])  # ships field

        counts.append(ships)
    return counts


def _seat_order_for_game(agent_specs, game_id):
    if len(agent_specs) == 2:
        if game_id % 2 == 0:
            return list(agent_specs)
        return [agent_specs[1], agent_specs[0]]

    rotation = game_id % len(agent_specs)
    return list(agent_specs[rotation:] + agent_specs[:rotation])


def _score_tuple(reward, ships):
    return (int(reward or 0), int(ships or 0))


def _best_seat(rewards, ships):
    ranks = [_score_tuple(rewards[i], ships[i]) for i in range(len(rewards))]
    best_rank = max(ranks)
    winners = [i for i, rank in enumerate(ranks) if rank == best_rank]
    return winners, best_rank


def _env_to_json_dict(env):
    replay_data = env.toJSON()
    if isinstance(replay_data, str):
        return json.loads(replay_data)
    return replay_data


def _save_replay(env, replay_dir, game_id, seat_order, game_result, mode):
    os.makedirs(replay_dir, exist_ok=True)
    replay_path = os.path.join(replay_dir, f"game_{game_id:03d}.json")
    try:
        replay_data = _env_to_json_dict(env)
        replay_data["agentNames"] = [spec.name for spec in seat_order]
        replay_data["agentRefs"] = [spec.raw for spec in seat_order]
        replay_data["matchMode"] = mode
        replay_data["gameResult"] = {
            "winner": game_result.get("winner"),
            "winnerIndex": game_result.get("winner_index"),
            "rewards": game_result.get("rewards", []),
            "shipCounts": game_result.get("ship_counts", []),
        }
        replay_data["telemetry"] = game_result.get("telemetry", {})
        replay_data["stats"] = game_result.get("stats", {})
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(replay_data, f, ensure_ascii=True)
        return replay_path, None
    except Exception as e:
        return None, str(e)


def run_game(agent_specs, game_id, save_replay=False, replay_dir="replays"):
    """Run a single game and return results."""
    if len(agent_specs) not in (2, 4):
        raise ValueError(f"Orbit Wars supports 2 or 4 agents, got {len(agent_specs)}")

    env = make("orbit_wars", debug=False)
    seat_order = _seat_order_for_game(agent_specs, game_id)
    agents = [spec.ref for spec in seat_order]
    seat_index_by_original = {spec.index: seat_idx for seat_idx, spec in enumerate(seat_order)}

    reset_telemetry()
    start = time.time()
    try:
        result = env.run(agents)
    except Exception as e:
        return {
            "game_id": game_id,
            "error": str(e),
            "steps": 0,
            "elapsed": time.time() - start,
            "seat_names": [spec.name for spec in seat_order],
            "seat_index_by_original": seat_index_by_original,
        }
    elapsed = time.time() - start

    final = result[-1]
    rewards = [final[i].get("reward", 0) or 0 for i in range(len(final))]
    statuses = [final[i].get("status", "DONE") for i in range(len(final))]
    ship_counts = get_final_ship_counts(result)
    winners, best_rank = _best_seat(rewards, ship_counts)
    winner_index = winners[0] if len(winners) == 1 else None
    winner_name = seat_order[winner_index].name if winner_index is not None else "TIE"
    steps = len(result)

    game_result = {
        "game_id": game_id,
        "seat_names": [spec.name for spec in seat_order],
        "seat_refs": [spec.raw for spec in seat_order],
        "seat_index_by_original": seat_index_by_original,
        "winner_index": winner_index,
        "winner": winner_name,
        "best_rank": best_rank,
        "rewards": rewards,
        "ship_counts": ship_counts,
        "steps": steps,
        "elapsed": elapsed,
        "statuses": statuses,
        "error": None,
        "telemetry": telemetry_snapshot(),
    }
    game_result["stats"] = summarize_game_stats(result, game_result, game_result["seat_names"])

    if save_replay:
        replay_path, replay_error = _save_replay(
            env,
            replay_dir,
            game_id,
            seat_order,
            game_result,
            mode="ffa" if len(agent_specs) == 4 else "match",
        )
        if replay_path:
            game_result["replay_path"] = replay_path
        if replay_error:
            game_result["replay_error"] = replay_error

    return game_result


def _write_stats_file(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _stats_payload(kind, results):
    if isinstance(results, dict) and "results" in results:
        games = results["results"]
        metadata = {key: value for key, value in results.items() if key != "results"}
    elif isinstance(results, list):
        games = []
        for item in results:
            if isinstance(item, dict) and "results" in item:
                games.extend(item["results"])
        metadata = {"matches": len(results)}
    else:
        games = []
        metadata = {}

    return {
        "kind": kind,
        "metadata": metadata,
        "aggregate": aggregate_game_stats(games),
        "games": [
            {
                "game_id": game.get("game_id"),
                "seat_names": game.get("seat_names", []),
                "winner": game.get("winner"),
                "rewards": game.get("rewards", []),
                "ship_counts": game.get("ship_counts", []),
                "steps": game.get("steps", 0),
                "elapsed": game.get("elapsed", 0.0),
                "telemetry": game.get("telemetry", {}),
                "stats": game.get("stats", {}),
                "error": game.get("error"),
            }
            for game in games
        ],
    }


def run_match(agent_a, agent_b, num_games, save_replays=False, verbose=True, name_a=None, name_b=None):
    """Run a full match between two agents."""
    if name_a is None:
        name_a = agent_a.name
    if name_b is None:
        name_b = agent_b.name

    if verbose:
        print(f"\n{'='*65}")
        print(f"  MATCH: {name_a} vs {name_b}  ({num_games} games)")
        print(f"{'='*65}")

    results = []
    a_wins = 0
    b_wins = 0
    ties = 0
    errors = 0
    total_a_ships = 0
    total_b_ships = 0

    for i in range(num_games):
        game = run_game([agent_a, agent_b], i, save_replays)
        results.append(game)

        if game["error"]:
            errors += 1
            if verbose:
                print(f"  Game {i+1:2d}: ERROR - {game['error'][:60]}")
            continue

        seat_a = game["seat_index_by_original"][agent_a.index]
        seat_b = game["seat_index_by_original"][agent_b.index]
        a_rank = _score_tuple(game["rewards"][seat_a], game["ship_counts"][seat_a])
        b_rank = _score_tuple(game["rewards"][seat_b], game["ship_counts"][seat_b])

        if a_rank > b_rank:
            a_wins += 1
            tag = f"{name_a} WIN"
        elif b_rank > a_rank:
            b_wins += 1
            tag = f"{name_b} WIN"
        else:
            ties += 1
            tag = "TIE"

        total_a_ships += game["ship_counts"][seat_a]
        total_b_ships += game["ship_counts"][seat_b]

        if verbose:
            margin = f"({game['ship_counts'][seat_a]:3d} vs {game['ship_counts'][seat_b]:3d})"
            pos = f"[{name_a} as P{seat_a}]"
            print(f"  Game {i+1:2d}: {tag:20s} {margin}  {game['steps']:3d} steps  {game['elapsed']:.1f}s  {pos}")

    valid = num_games - errors
    if verbose:
        print(f"\n{'-'*65}")
        print(f"  RESULTS ({valid} valid games):")
        print(f"    {name_a:20s}  Wins: {a_wins:3d}  ({a_wins/max(1,valid)*100:.0f}%)")
        print(f"    {name_b:20s}  Wins: {b_wins:3d}  ({b_wins/max(1,valid)*100:.0f}%)")
        print(f"    {'Ties':20s}       {ties:3d}  ({ties/max(1,valid)*100:.0f}%)")
        if valid > 0:
            avg_a = total_a_ships / valid
            avg_b = total_b_ships / valid
            print(f"\n    Avg ships at end:  {name_a}={avg_a:.0f}  {name_b}={avg_b:.0f}  (margin: {avg_a-avg_b:+.0f})")
        if errors > 0:
            print(f"    Errors: {errors}")
        print(f"{'='*65}\n")

    return {
        "agent_a": name_a,
        "agent_b": name_b,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "errors": errors,
        "results": results,
    }


def run_tournament(agent_paths, games_per_match, save_replays=False):
    """Round-robin tournament between multiple agents."""
    n = len(agent_paths)
    names = [spec.name for spec in agent_paths]

    print(f"\n{'#'*65}")
    print(f"  TOURNAMENT: {n} agents, {games_per_match} games per match")
    print(f"  Agents: {', '.join(names)}")
    print(f"{'#'*65}")

    wins = defaultdict(int)
    losses = defaultdict(int)
    played = defaultdict(int)
    match_results = []

    for i in range(n):
        for j in range(i + 1, n):
            result = run_match(agent_paths[i], agent_paths[j], games_per_match, save_replays)
            match_results.append(result)

            wins[names[i]] += result["a_wins"]
            losses[names[i]] += result["b_wins"]
            played[names[i]] += games_per_match - result["errors"]

            wins[names[j]] += result["b_wins"]
            losses[names[j]] += result["a_wins"]
            played[names[j]] += games_per_match - result["errors"]

    # Print standings
    print(f"\n{'='*65}")
    print(f"  TOURNAMENT STANDINGS")
    print(f"{'-'*65}")
    print(f"  {'Agent':25s} {'W':>4s} {'L':>4s} {'P':>4s} {'Win%':>6s}")
    print(f"{'-'*65}")

    standings = sorted(names, key=lambda n: wins[n] / max(1, played[n]), reverse=True)
    for name in standings:
        w = wins[name]
        l = losses[name]
        p = played[name]
        rate = w / max(1, p) * 100
        print(f"  {name:25s} {w:4d} {l:4d} {p:4d} {rate:5.1f}%")

    print(f"{'='*65}\n")
    return match_results


def run_ffa(agent_paths, games_per_match, save_replays=False):
    """Run a 4-player free-for-all match."""
    if len(agent_paths) != 4:
        raise ValueError(f"FFA mode requires exactly 4 agents, got {len(agent_paths)}")

    names = [spec.name for spec in agent_paths]
    print(f"\n{'='*65}")
    print(f"  FFA: {' vs '.join(names)}  ({games_per_match} games)")
    print(f"{'='*65}")

    wins = defaultdict(int)
    ties = 0
    errors = 0
    results = []

    for i in range(games_per_match):
        game = run_game(agent_paths, i, save_replays)
        results.append(game)

        if game["error"]:
            errors += 1
            print(f"  Game {i+1:2d}: ERROR - {game['error'][:60]}")
            continue

        winner = game["winner"]
        if winner == "TIE":
            ties += 1
            tag = "TIE"
        else:
            wins[winner] += 1
            tag = f"{winner} WIN"

        if game["winner_index"] is None:
            rank_label = "tie"
        else:
            rank_label = f"P{game['winner_index']}"
        print(
            f"  Game {i+1:2d}: {tag:20s}  "
            f"{game['ship_counts']}  {game['steps']:3d} steps  {game['elapsed']:.1f}s  "
            f"[{rank_label}]"
        )

    valid = games_per_match - errors
    print(f"\n{'-'*65}")
    print(f"  RESULTS ({valid} valid games):")
    for name in names:
        w = wins[name]
        print(f"    {name:20s}  Wins: {w:3d}  ({w/max(1, valid)*100:.0f}%)")
    print(f"    {'Ties':20s}       {ties:3d}  ({ties/max(1, valid)*100:.0f}%)")
    if errors > 0:
        print(f"    Errors: {errors}")
    print(f"{'='*65}\n")

    return {
        "agents": names,
        "wins": dict(wins),
        "ties": ties,
        "errors": errors,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orbit Wars Local Arena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python arena.py submission.py submission_958_raw.py --games 10
  python arena.py submission.py submission_958_raw.py --games 5 --save-replays
  python arena.py --tournament submission.py submission_958_raw.py agent3.py --games 4
  python arena.py --ffa main.py starter starter starter --games 10
        """,
    )
    parser.add_argument("agents", nargs="+", help="Agent Python files to test")
    parser.add_argument("--games", type=int, default=10, help="Games per match (default: 10)")
    parser.add_argument("--save-replays", action="store_true", help="Save game replays as JSON")
    parser.add_argument("--tournament", action="store_true", help="Run round-robin tournament")
    parser.add_argument("--ffa", action="store_true", help="Run a 4-player free-for-all match")
    parser.add_argument("--stats-file", help="Write per-game telemetry and replay-derived stats to JSON")

    args = parser.parse_args()

    # Resolve agents - supports .py, .ipynb, and built-in aliases like starter.
    resolved = resolve_agent_specs(args.agents)

    if args.ffa:
        result = run_ffa(resolved, args.games, args.save_replays)
        result_kind = "ffa"
    elif args.tournament or len(resolved) > 2:
        result = run_tournament(resolved, args.games, args.save_replays)
        result_kind = "tournament"
    elif len(resolved) == 2:
        result = run_match(resolved[0], resolved[1], args.games, args.save_replays)
        result_kind = "match"
    else:
        print("ERROR: Need at least 2 agents for a match")
        sys.exit(1)

    if args.stats_file:
        _write_stats_file(args.stats_file, _stats_payload(result_kind, result))
        print(f"Stats written to {args.stats_file}")
