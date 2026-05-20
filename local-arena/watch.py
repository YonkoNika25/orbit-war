"""
Generate replay data from a game and launch the visual arena viewer.

Usage:
  python watch.py submission.py submission_958_raw.py
  python watch.py submission.py submission_958_raw.py --speed 2
  python watch.py submission.py submission_958_raw.py --games 5 --output-dir watch_replays
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from kaggle_environments import make

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.telemetry import reset as reset_telemetry, snapshot as telemetry_snapshot


def _unique_names(names):
    counts = defaultdict(int)
    unique = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            unique.append(name)
        else:
            unique.append(f"{name} ({counts[name]})")
    return unique


def _seat_order_for_game(agents, agent_names, game_id):
    if len(agents) == 2:
        if game_id % 2 == 0:
            return agents, agent_names
        return agents[::-1], agent_names[::-1]

    rotation = game_id % len(agents)
    return (
        agents[rotation:] + agents[:rotation],
        agent_names[rotation:] + agent_names[:rotation],
    )


def run_and_capture(agents, agent_names, configuration=None):
    """Run a game and capture full step-by-step data for visualization."""
    env = make("orbit_wars", configuration=dict(configuration or {}), debug=False)
    reset_telemetry()
    result = env.run(agents)

    # Extract configuration
    config = {
        "boardSize": 100,
        "sunX": 50,
        "sunY": 50,
        "sunRadius": 10,
        "totalSteps": 500,
        "shipSpeed": env.configuration.get("shipSpeed", 6.0) if hasattr(env.configuration, "get") else 6.0,
    }

    # Extract per-step state
    frames = []
    for step_idx, step_data in enumerate(result):
        frame = {"step": step_idx, "planets": [], "fleets": [], "scores": []}

        for player_idx in range(len(step_data)):
            agent_data = step_data[player_idx]
            obs = agent_data.get("observation")
            reward = agent_data.get("reward")
            status = agent_data.get("status", "ACTIVE")

            if obs is None:
                frame["scores"].append({
                    "player": player_idx,
                    "ships": 0,
                    "planets": 0,
                    "status": status,
                    "reward": reward,
                })
                continue

            raw_planets = obs.get("planets", []) or []
            raw_fleets = obs.get("fleets", []) or []

            # Only add planets/fleets from player 0's view (same data for all)
            if player_idx == 0:
                for p in raw_planets:
                    frame["planets"].append({
                        "id": p[0], "owner": p[1],
                        "x": p[2], "y": p[3],
                        "radius": p[4], "ships": p[5],
                        "production": p[6],
                    })
                for f in raw_fleets:
                    frame["fleets"].append({
                        "id": f[0], "owner": f[1],
                        "x": f[2], "y": f[3],
                        "angle": f[4], "from_planet_id": f[5],
                        "ships": f[6],
                    })

            # Calculate score
            player_ships = 0
            player_planets = 0
            for p in raw_planets:
                if p[1] == player_idx:
                    player_ships += int(p[5])
                    player_planets += 1
            for f in raw_fleets:
                if f[1] == player_idx:
                    player_ships += int(f[6])

            frame["scores"].append({
                "player": player_idx,
                "ships": player_ships,
                "planets": player_planets,
                "status": status,
                "reward": reward,
            })

        frames.append(frame)

    return {
        "config": config,
        "frames": frames,
        "numPlayers": len(result[0]),
        "agentNames": agent_names,
        "telemetry": _json_safe(telemetry_snapshot()),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value


def _render_standalone_html(replay_data, title):
    base_dir = Path(__file__).resolve().parent
    viewer_path = base_dir / "viewer.html"
    template = viewer_path.read_text(encoding="utf-8")
    payload = json.dumps(_json_safe(replay_data), ensure_ascii=True).replace("</", "<\\/")
    script = f"<script>const REPLAY_DATA = JSON.parse({json.dumps(payload)});</script>"
    if '<script src="replay_data.js"></script>' in template:
        template = template.replace('<script src="replay_data.js"></script>', script, 1)
    else:
        template = template.replace("<script src=\"replay_data.js\"></script>", script, 1)
    if "<title>Orbit Wars Replay</title>" in template:
        template = template.replace("<title>Orbit Wars Replay</title>", f"<title>{title}</title>", 1)
    return template


def save_game_html(replay_data, output_dir, game_id, agent_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_names = "_vs_".join(name.replace(" ", "_") for name in agent_names)
    html_path = output_dir / f"game_{game_id:03d}_{safe_names}.html"
    html = _render_standalone_html(replay_data, title=f"Orbit Wars - {safe_names}")
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="Watch Orbit Wars games live",
        epilog="  python watch.py agent1.ipynb agent2.ipynb\n  python watch.py submission.py agent.ipynb --speed 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("agents", nargs="+", help="Agent files (.py or .ipynb) or built-in aliases")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port")
    parser.add_argument("--games", type=int, default=1, help="Number of games to run and save")
    parser.add_argument("--seed", type=int, help="Optional Orbit Wars environment seed")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the HTML replay")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_replays"),
        help="Directory to save HTML replays",
    )
    args = parser.parse_args()

    if len(args.agents) not in (2, 4):
        print("ERROR: Orbit Wars watch mode supports 2 or 4 agents.")
        sys.exit(1)

    # Resolve agents (supports .ipynb and built-in aliases like starter).
    from notebook_util import resolve_agent_path
    agents = []
    names = []
    for raw in args.agents:
        resolved, name = resolve_agent_path(raw)
        agents.append(resolved)
        names.append(name)

    unique_names = _unique_names(names)
    saved_paths = []
    total_start = time.time()
    configuration = {"seed": args.seed} if args.seed is not None else None
    for game_id in range(args.games):
        seat_agents, seat_names = _seat_order_for_game(agents, unique_names, game_id)
        print(f"Running game {game_id + 1}/{args.games}: {' vs '.join(seat_names)}...")
        start = time.time()
        replay_data = run_and_capture(seat_agents, seat_names, configuration=configuration)
        elapsed = time.time() - start
        print(f"Game {game_id + 1} completed in {elapsed:.1f}s ({len(replay_data['frames'])} frames)")

        replay_data["agentNames"] = seat_names
        replay_data["speed"] = args.speed
        replay_data["gameId"] = game_id

        html_path = save_game_html(replay_data, args.output_dir, game_id, seat_names)
        saved_paths.append(str(html_path))
        print(f"Saved: {html_path}")

    total_elapsed = time.time() - total_start
    print(f"Saved {len(saved_paths)} HTML replay(s) in {total_elapsed:.1f}s")

    if args.games == 1 and saved_paths and not args.no_open:
        url = "file:///" + saved_paths[0].replace("\\", "/")
        print(f"Opening viewer: {url}")
        import webbrowser

        webbrowser.open(url)


if __name__ == "__main__":
    main()
