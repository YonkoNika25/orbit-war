from __future__ import annotations

import argparse
from pathlib import Path

from notebook_util import resolve_agent_path
from watch import run_and_capture, save_game_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a visual Epic 1-3 demo replay.")
    parser.add_argument("--seed", type=int, default=7, help="Orbit Wars environment seed")
    parser.add_argument("--opponent", default="random", help="Opponent agent path or built-in alias")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "watch_replays"),
        help="Directory to save the standalone HTML replay",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    demo_agent_path = base_dir / "demo_epic3_agent.py"
    demo_ref, demo_name = resolve_agent_path(str(demo_agent_path))
    opponent_ref, opponent_name = resolve_agent_path(args.opponent)

    replay_data = run_and_capture(
        [demo_ref, opponent_ref],
        [demo_name, opponent_name],
        configuration={"seed": args.seed},
    )
    replay_data["speed"] = 1.0
    replay_data["gameId"] = 0
    html_path = save_game_html(replay_data, args.output_dir, 0, [demo_name, opponent_name])
    print(html_path)


if __name__ == "__main__":
    main()
