from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kaggle_environments import make

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.actions import to_kaggle_action
from core.geometry import (
    angle_to,
    estimate_eta,
    find_intercept_angle,
    fleet_speed_for_ships,
    predict_position,
)
from core.parser import parse_observation
from core.telemetry import record_step, reset as reset_telemetry, snapshot as telemetry_snapshot
from core.types import Planet
from watch import save_game_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-shot Orbit Wars debug scenario.")
    parser.add_argument("--seed", type=int, default=7, help="Orbit Wars environment seed")
    parser.add_argument("--source-id", type=int, help="Owned planet id to fire from")
    parser.add_argument("--target-id", type=int, help="Target planet id to fire at")
    parser.add_argument("--ships", type=int, help="Ships to launch")
    parser.add_argument("--fire-step", type=int, default=0, help="Turn to fire on")
    parser.add_argument("--wait-steps", type=int, default=0, help="Advance the environment before selecting source/target and firing")
    parser.add_argument(
        "--angle-mode",
        choices=("static", "intercept"),
        default="static",
        help="How to compute the launch angle",
    )
    parser.add_argument("--list-planets", action="store_true", help="Print planets for this seed and exit")
    parser.add_argument("--list-step", type=int, default=0, help="Environment step to inspect when using --list-planets")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "watch_replays"),
        help="Directory to save the HTML replay and JSON report",
    )
    args = parser.parse_args()

    if args.list_planets:
        list_planets(args.seed, args.list_step)
        return

    if args.source_id is None or args.target_id is None:
        raise SystemExit("source-id and target-id are required unless --list-planets is used")

    replay_data, report = run_single_shot(
        seed=args.seed,
        source_id=args.source_id,
        target_id=args.target_id,
        ships=args.ships,
        fire_step=args.fire_step,
        wait_steps=args.wait_steps,
        angle_mode=args.angle_mode,
    )

    output_dir = Path(args.output_dir)
    html_path = save_game_html(replay_data, output_dir, 0, ["single_shot", "noop"])
    report_path = output_dir / "single_shot_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print_summary(report)
    print(html_path)
    print(report_path)


def list_planets(seed: int, list_step: int) -> None:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    state = env.reset()
    for _ in range(max(0, int(list_step))):
        state = env.step([[], []])
    parsed = parse_observation(state[0].observation, env.configuration)
    print(f"Seed: {seed}")
    print(f"Step: {parsed.step}")
    print("Planets:")
    for planet in sorted(parsed.planets, key=lambda item: item.id):
        print(
            {
                "id": planet.id,
                "owner": planet.owner,
                "ships": planet.ships,
                "production": planet.production,
                "x": round(planet.x, 3),
                "y": round(planet.y, 3),
                "moving": bool(planet.extras.get("is_moving")),
            }
        )


def run_single_shot(
    *,
    seed: int,
    source_id: int,
    target_id: int,
    ships: int | None,
    fire_step: int,
    wait_steps: int,
    angle_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    reset_telemetry()
    adjusted_fire_step = max(0, int(fire_step)) + max(0, int(wait_steps))

    scenario: Dict[str, Any] = {}

    def single_shot_agent(obs: Any, config: Any) -> List[List[float | int]]:
        parsed = parse_observation(obs, config)
        if parsed.step != adjusted_fire_step:
            record_step(parsed.step, parsed.player_id, _idle_telemetry(parsed, reason="waiting_for_fire_step"))
            return []

        source = next((planet for planet in parsed.planets if planet.id == source_id), None)
        target = next((planet for planet in parsed.planets if planet.id == target_id), None)
        if source is None or target is None:
            raise ValueError(f"source_id={source_id} or target_id={target_id} missing on step {parsed.step}")
        if source.owner != parsed.player_id:
            raise ValueError(f"source_id={source_id} is not owned by player {parsed.player_id} on step {parsed.step}")

        selected_ships = max(1, min(int(ships if ships is not None else source.ships), int(source.ships)))
        static_angle = angle_to((source.x, source.y), (target.x, target.y))
        speed = fleet_speed_for_ships(selected_ships, max_speed=float(config.shipSpeed))
        intercept = find_intercept_angle(source, target, speed, max_steps=150)
        selected_angle = static_angle
        if angle_mode == "intercept" and intercept is not None:
            selected_angle = intercept[0]

        action = to_kaggle_action(source.id, selected_angle, selected_ships)
        scenario.update(
            {
                "seed": seed,
                "fire_step": adjusted_fire_step,
                "requested_fire_step": fire_step,
                "wait_steps": wait_steps,
                "source": _planet_payload(source),
                "target": _planet_payload(target),
                "ships": selected_ships,
                "angle_mode": angle_mode,
                "selected_angle": selected_angle,
                "static_angle": static_angle,
                "intercept_angle": None if intercept is None else intercept[0],
                "intercept_step": None if intercept is None else intercept[1],
                "fleet_speed": speed,
                "true_eta": estimate_eta(source, target, speed=speed),
            }
        )
        record_step(
            parsed.step,
            parsed.player_id,
            {
                "step": parsed.step,
                "player_id": parsed.player_id,
                "summary": {
                    "my_planets": len(parsed.my_planets),
                    "enemy_planets": len(parsed.enemy_planets),
                    "neutral_planets": len(parsed.neutral_planets),
                    "my_fleets": sum(1 for fleet in parsed.fleets if fleet.owner == parsed.player_id),
                    "enemy_fleets": sum(1 for fleet in parsed.fleets if fleet.owner not in (-1, parsed.player_id)),
                    "remaining_overage_time": parsed.extras.get("remainingOverageTime"),
                },
                "global_features": {},
                "passes": [
                    {
                        "pass_index": 0,
                        "candidate_count": 1,
                        "legal_count": 1,
                        "source_commitments_before": {},
                        "selected_index": 0,
                        "selected_candidate": {
                            "index": 0,
                            "type": f"SINGLE_SHOT_{angle_mode.upper()}",
                            "legal": True,
                            "reject_reason": None,
                            "source_id": source.id,
                            "target_id": target.id,
                            "ships": selected_ships,
                            "eta": scenario["true_eta"],
                            "angle": selected_angle,
                            "score": None,
                            "estimated_owner": target.owner,
                            "estimated_ships": target.ships,
                            "source": _planet_payload(source),
                            "target": _planet_payload(target),
                            "features": {
                                "target_is_moving": 1.0 if target.extras.get("is_moving") else 0.0,
                                "fleet_speed": speed,
                                "static_angle": static_angle,
                                "intercept_angle": 0.0 if intercept is None else intercept[0],
                            },
                            "label": f"SINGLE_SHOT {source.id}->{target.id} ships={selected_ships}",
                        },
                        "selected_action": list(action),
                        "applied": True,
                        "candidates": [],
                    }
                ],
                "final_actions": [list(action)],
            },
        )
        return [action]

    def noop_agent(obs: Any, config: Any) -> List[Any]:
        return []

    result = env.run([single_shot_agent, noop_agent])
    replay_data = capture_replay(env, result, ["single_shot", "noop"])
    report = build_single_shot_report(result, replay_data, telemetry_snapshot(), scenario)
    return replay_data, report


def capture_replay(env: Any, result: List[Any], agent_names: List[str]) -> Dict[str, Any]:
    frames = []
    for step_idx, step_data in enumerate(result):
        frame = {"step": step_idx, "planets": [], "fleets": [], "scores": []}
        for player_idx, agent_data in enumerate(step_data):
            obs = agent_data.get("observation")
            reward = agent_data.get("reward")
            status = agent_data.get("status", "ACTIVE")
            if obs is None:
                frame["scores"].append({"player": player_idx, "ships": 0, "planets": 0, "status": status, "reward": reward})
                continue

            if player_idx == 0:
                for p in obs.get("planets", []) or []:
                    frame["planets"].append(
                        {
                            "id": p[0],
                            "owner": p[1],
                            "x": p[2],
                            "y": p[3],
                            "radius": p[4],
                            "ships": p[5],
                            "production": p[6],
                        }
                    )
                for f in obs.get("fleets", []) or []:
                    frame["fleets"].append(
                        {
                            "id": f[0],
                            "owner": f[1],
                            "x": f[2],
                            "y": f[3],
                            "angle": f[4],
                            "from_planet_id": f[5],
                            "ships": f[6],
                        }
                    )

            ships = sum(int(p[5]) for p in (obs.get("planets", []) or []) if p[1] == player_idx)
            ships += sum(int(f[6]) for f in (obs.get("fleets", []) or []) if f[1] == player_idx)
            planets = sum(1 for p in (obs.get("planets", []) or []) if p[1] == player_idx)
            frame["scores"].append({"player": player_idx, "ships": ships, "planets": planets, "status": status, "reward": reward})
        frames.append(frame)

    return {
        "config": {
            "boardSize": 100,
            "sunX": 50,
            "sunY": 50,
            "sunRadius": 10,
            "totalSteps": 500,
            "shipSpeed": env.configuration.get("shipSpeed", 6.0) if hasattr(env.configuration, "get") else 6.0,
        },
        "frames": frames,
        "numPlayers": len(result[0]),
        "agentNames": agent_names,
        "telemetry": telemetry_snapshot(),
        "speed": 1.0,
        "gameId": 0,
    }


def build_single_shot_report(
    result: List[Any],
    replay_data: Dict[str, Any],
    telemetry: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    source = scenario["source"]
    target = scenario["target"]
    speed = float(scenario["fleet_speed"])
    angle = float(scenario["selected_angle"])
    fire_step = int(scenario["fire_step"])
    source_id = int(source["id"])
    ships = int(scenario["ships"])

    frames = replay_data["frames"]
    launch_frame_index = fire_step + 1
    launched_fleet_id = None
    actual_trace = []
    disappearance = None

    for frame in frames[launch_frame_index:]:
        matching = [
            fleet
            for fleet in frame["fleets"]
            if fleet["owner"] == 0 and fleet["from_planet_id"] == source_id and fleet["ships"] == ships
        ]
        if launched_fleet_id is None and matching:
            launched_fleet_id = matching[0]["id"]
        fleet = next((item for item in matching if item["id"] == launched_fleet_id), None)
        if fleet is None:
            if launched_fleet_id is not None:
                disappearance = classify_disappearance(actual_trace[-1] if actual_trace else None, frame, target["id"])
            break

        tick = frame["step"] - fire_step
        expected_x, expected_y = expected_position(source, angle, speed, tick)
        actual_trace.append(
            {
                "frame_step": frame["step"],
                "flight_tick": tick,
                "fleet_id": fleet["id"],
                "actual_x": fleet["x"],
                "actual_y": fleet["y"],
                "expected_x": expected_x,
                "expected_y": expected_y,
                "delta": math.hypot(fleet["x"] - expected_x, fleet["y"] - expected_y),
                "target_x": target_position_for_tick(target, tick)[0],
                "target_y": target_position_for_tick(target, tick)[1],
            }
        )

    if launched_fleet_id is None:
        disappearance = {"kind": "no_launch_observed"}

    final = result[-1]
    return {
        "scenario": scenario,
        "telemetry_counters": telemetry.get("counters", {}),
        "launch_frame_index": launch_frame_index,
        "launched_fleet_id": launched_fleet_id,
        "trace_length": len(actual_trace),
        "trace": actual_trace,
        "disappearance": disappearance,
        "final": {
            "statuses": [item.status for item in final],
            "rewards": [item.reward for item in final],
            "steps": len(result),
        },
    }


def _idle_telemetry(parsed: Any, reason: str) -> Dict[str, Any]:
    return {
        "step": int(parsed.step),
        "player_id": int(parsed.player_id),
        "summary": {
            "my_planets": len(parsed.my_planets),
            "enemy_planets": len(parsed.enemy_planets),
            "neutral_planets": len(parsed.neutral_planets),
            "my_fleets": sum(1 for fleet in parsed.fleets if fleet.owner == parsed.player_id),
            "enemy_fleets": sum(1 for fleet in parsed.fleets if fleet.owner not in (-1, parsed.player_id)),
            "remaining_overage_time": parsed.extras.get("remainingOverageTime"),
            "idle_reason": reason,
        },
        "global_features": {},
        "passes": [],
        "final_actions": [],
    }


def _planet_payload(planet: Planet) -> Dict[str, Any]:
    return {
        "id": int(planet.id),
        "owner": int(planet.owner),
        "ships": int(planet.ships),
        "production": float(planet.production),
        "x": float(planet.x),
        "y": float(planet.y),
        "radius": float(planet.radius),
        "orbit_center_x": float(planet.orbit_center_x),
        "orbit_center_y": float(planet.orbit_center_y),
        "orbit_radius": float(planet.orbit_radius),
        "orbit_angle": float(planet.orbit_angle),
        "orbit_speed": float(planet.orbit_speed),
        "is_moving": bool(planet.extras.get("is_moving")),
    }


def target_position_for_tick(target_payload: Dict[str, Any], tick: int) -> Tuple[float, float]:
    planet = Planet(
        id=int(target_payload["id"]),
        owner=int(target_payload["owner"]),
        ships=int(target_payload["ships"]),
        production=float(target_payload["production"]),
        x=float(target_payload["x"]),
        y=float(target_payload["y"]),
        radius=float(target_payload["radius"]),
        orbit_center_x=float(target_payload.get("orbit_center_x", target_payload["x"])),
        orbit_center_y=float(target_payload.get("orbit_center_y", target_payload["y"])),
        orbit_radius=float(target_payload.get("orbit_radius", 0.0)),
        orbit_angle=float(target_payload.get("orbit_angle", 0.0)),
        orbit_speed=float(target_payload.get("orbit_speed", 0.0)),
        extras={"is_moving": bool(target_payload.get("is_moving"))},
    )
    return predict_position(planet, tick)


def expected_position(source_payload: Dict[str, Any], angle: float, speed: float, tick: int) -> Tuple[float, float]:
    travel = float(source_payload["radius"]) + 0.1 + speed * tick
    return (
        float(source_payload["x"]) + math.cos(angle) * travel,
        float(source_payload["y"]) + math.sin(angle) * travel,
    )


def classify_disappearance(last_trace: Dict[str, Any] | None, frame: Dict[str, Any], target_id: int) -> Dict[str, Any]:
    if last_trace is None:
        return {"kind": "unknown"}
    target = next((planet for planet in frame["planets"] if int(planet["id"]) == int(target_id)), None)
    if target is not None:
        return {"kind": "planet_or_target_resolution", "target_owner": target["owner"], "target_ships": target["ships"]}
    if not (0.0 <= last_trace["actual_x"] <= 100.0 and 0.0 <= last_trace["actual_y"] <= 100.0):
        return {"kind": "out_of_bounds"}
    return {"kind": "unknown"}


def print_summary(report: Dict[str, Any]) -> None:
    scenario = report["scenario"]
    print("Scenario:")
    print(
        {
            "seed": scenario["seed"],
            "source_id": scenario["source"]["id"],
            "target_id": scenario["target"]["id"],
            "ships": scenario["ships"],
            "fire_step": scenario["fire_step"],
            "requested_fire_step": scenario.get("requested_fire_step"),
            "wait_steps": scenario.get("wait_steps"),
            "angle_mode": scenario["angle_mode"],
            "selected_angle": scenario["selected_angle"],
            "static_angle": scenario["static_angle"],
            "intercept_angle": scenario["intercept_angle"],
            "true_eta": scenario["true_eta"],
            "target_is_moving": scenario["target"]["is_moving"],
        }
    )
    print("Trace summary:")
    print(
        {
            "launched_fleet_id": report["launched_fleet_id"],
            "trace_length": report["trace_length"],
            "disappearance": report["disappearance"],
            "final": report["final"],
        }
    )
    if report["trace"]:
        print("First ticks:")
        for row in report["trace"][: min(8, len(report["trace"]))]:
            print(
                {
                    "tick": row["flight_tick"],
                    "actual": (round(row["actual_x"], 3), round(row["actual_y"], 3)),
                    "expected": (round(row["expected_x"], 3), round(row["expected_y"], 3)),
                    "delta": round(row["delta"], 6),
                    "target": (round(row["target_x"], 3), round(row["target_y"], 3)),
                }
            )


if __name__ == "__main__":
    main()
