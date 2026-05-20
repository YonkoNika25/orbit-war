from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from kaggle_environments import make

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.geometry import angle_to, estimate_eta, find_intercept_angle, fleet_speed_for_ships
from core.telemetry import reset as reset_telemetry, snapshot as telemetry_snapshot
from core.types import Planet
from demo_epic3_agent import agent as demo_agent
from notebook_util import resolve_agent_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Epic 1-3 demo agent decisions against the real Kaggle environment.")
    parser.add_argument("--seed", type=int, default=7, help="Orbit Wars environment seed")
    parser.add_argument("--opponent", default="random", help="Opponent agent path or built-in alias")
    parser.add_argument("--top", type=int, default=25, help="How many suspicious selections to print")
    parser.add_argument(
        "--report",
        default=str(Path(__file__).resolve().parent / "watch_replays" / "debug_epic3_report.json"),
        help="Where to write the JSON report",
    )
    args = parser.parse_args()

    opponent_ref, opponent_name = resolve_agent_path(args.opponent)

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
    reset_telemetry()
    result = env.run([demo_agent, opponent_ref])
    telemetry = telemetry_snapshot()

    report = analyze_run(telemetry, env.configuration, opponent_name, args.seed, result)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print_summary(report, args.top)
    print(report_path)


def analyze_run(
    telemetry: Dict[str, Any],
    config: Any,
    opponent_name: str,
    seed: int,
    result: List[Any],
) -> Dict[str, Any]:
    max_speed = float(getattr(config, "shipSpeed", 6.0))
    steps = telemetry.get("steps", [])
    selected_rows: List[Dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    selected_type_counts: Counter[str] = Counter()

    for step_entry in steps:
        for candidate in _iter_candidates(step_entry):
            if not candidate.get("legal", False):
                rejection_counts[str(candidate.get("reject_reason") or "unknown")] += 1

        for pass_entry in step_entry.get("passes", []):
            selected = pass_entry.get("selected_candidate")
            if not selected:
                continue
            analysis = analyze_selection(step_entry, pass_entry, max_speed)
            selected_rows.append(analysis)
            selected_type_counts[analysis["selected_type"]] += 1

    final = result[-1]
    final_obs = final[0].observation
    owned_planets = [planet for planet in final_obs.planets if planet[1] == 0]
    owned_fleets = [fleet for fleet in final_obs.fleets if fleet[1] == 0]

    suspicious = sorted(
        [row for row in selected_rows if row["risk_flags"]],
        key=lambda row: (-len(row["risk_flags"]), -abs(row["eta_error"]), row["step"], row["pass_index"]),
    )

    return {
        "seed": seed,
        "opponent": opponent_name,
        "counters": telemetry.get("counters", {}),
        "selected_type_counts": dict(selected_type_counts),
        "rejection_counts": dict(rejection_counts),
        "final": {
            "statuses": [item.status for item in final],
            "rewards": [item.reward for item in final],
            "owned_planets": len(owned_planets),
            "owned_fleets": len(owned_fleets),
            "owned_total_ships": sum(int(planet[5]) for planet in owned_planets) + sum(int(fleet[6]) for fleet in owned_fleets),
            "steps": len(result),
        },
        "selected": selected_rows,
        "suspicious": suspicious,
    }


def analyze_selection(step_entry: Dict[str, Any], pass_entry: Dict[str, Any], max_speed: float) -> Dict[str, Any]:
    candidate = pass_entry["selected_candidate"]
    source = candidate.get("source")
    target = candidate.get("target")
    ships = int(candidate.get("ships") or 0)
    chosen_eta = int(candidate.get("eta") or 0)
    selected_action = list(pass_entry.get("selected_action") or [])
    selected_type = str(candidate.get("type"))
    is_launch = bool(selected_action)

    true_speed = fleet_speed_for_ships(ships, max_speed=max_speed) if ships > 0 else 0.0
    true_eta = chosen_eta
    exit_tick = None
    intercept = None
    angle_error = None
    static_angle = None
    moving_target = bool(target and target.get("is_moving"))
    if source and target and ships > 0:
        source_planet = _planet_from_payload(source)
        target_planet = _planet_from_payload(target)
        true_eta = estimate_eta(source_planet, target_planet, speed=true_speed)
        exit_tick = _exit_tick(source_planet, float(candidate.get("angle") or 0.0), true_speed)
        static_angle = angle_to((source_planet.x, source_planet.y), (target_planet.x, target_planet.y))
        if moving_target:
            intercept = find_intercept_angle(source_planet, target_planet, true_speed, max_steps=min(150, max(20, true_eta * 2)))
            if intercept is not None:
                angle_error = _angle_delta(float(candidate.get("angle") or 0.0), intercept[0])

    risk_flags: List[str] = []
    if selected_type in {"STOP", "HOLD_SOURCE"}:
        risk_flags.append("no_launch_selected")
    if is_launch and abs(chosen_eta - true_eta) >= 2:
        risk_flags.append(f"eta_mismatch:{chosen_eta}->{true_eta}")
    if moving_target:
        risk_flags.append("moving_target_static_aim")
    if angle_error is not None and angle_error > 0.1:
        risk_flags.append(f"intercept_angle_delta:{angle_error:.3f}")
    if exit_tick is not None:
        risk_flags.append(f"exits_board_tick:{exit_tick}")
    if candidate.get("estimated_owner") != step_entry.get("player_id") and selected_type in {"ATTACK", "EXPAND_NEUTRAL"}:
        risk_flags.append("non_capturing_launch")

    return {
        "step": int(step_entry["step"]),
        "pass_index": int(pass_entry["pass_index"]),
        "selected_type": selected_type,
        "selected_action": selected_action,
        "ships": ships,
        "chosen_eta": chosen_eta,
        "true_eta": int(true_eta),
        "eta_error": int(chosen_eta - true_eta),
        "true_speed": true_speed,
        "moving_target": moving_target,
        "static_angle": static_angle,
        "selected_angle": float(candidate.get("angle") or 0.0),
        "intercept_angle": None if intercept is None else float(intercept[0]),
        "intercept_step": None if intercept is None else int(intercept[1]),
        "intercept_angle_delta": angle_error,
        "exit_tick": exit_tick,
        "risk_flags": risk_flags,
        "source_id": candidate.get("source_id"),
        "target_id": candidate.get("target_id"),
        "reject_reason": candidate.get("reject_reason"),
        "source_commitments_before": dict(pass_entry.get("source_commitments_before") or {}),
    }


def _iter_candidates(step_entry: Dict[str, Any]):
    for pass_entry in step_entry.get("passes", []):
        for candidate in pass_entry.get("candidates", []):
            yield candidate


def _planet_from_payload(payload: Dict[str, Any]) -> Planet:
    return Planet(
        id=int(payload["id"]),
        owner=int(payload["owner"]),
        ships=int(payload["ships"]),
        production=float(payload["production"]),
        x=float(payload["x"]),
        y=float(payload["y"]),
        radius=float(payload["radius"]),
        orbit_center_x=float(payload.get("orbit_center_x", payload["x"])),
        orbit_center_y=float(payload.get("orbit_center_y", payload["y"])),
        orbit_radius=float(payload.get("orbit_radius", 0.0)),
        orbit_angle=float(payload.get("orbit_angle", 0.0)),
        orbit_speed=float(payload.get("orbit_speed", 0.0)),
        extras={"is_moving": bool(payload.get("is_moving"))},
    )


def _exit_tick(source: Planet, angle: float, speed: float, board_min: float = 0.0, board_max: float = 100.0) -> int | None:
    if speed <= 0.0:
        return None

    x = source.x + math.cos(angle) * (source.radius + 0.1)
    y = source.y + math.sin(angle) * (source.radius + 0.1)
    for tick in range(1, 251):
        x += math.cos(angle) * speed
        y += math.sin(angle) * speed
        if x < board_min or x > board_max or y < board_min or y > board_max:
            return tick
    return None


def _angle_delta(a: float, b: float) -> float:
    delta = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def print_summary(report: Dict[str, Any], top: int) -> None:
    final = report["final"]
    print("Final:", final)
    print("Counters:", report["counters"])
    print("Selected types:", report["selected_type_counts"])
    print("Reject reasons:", report["rejection_counts"])
    print(f"Top suspicious selections ({min(top, len(report['suspicious']))} shown):")
    for row in report["suspicious"][:top]:
        print(
            {
                "step": row["step"],
                "pass": row["pass_index"],
                "type": row["selected_type"],
                "source": row["source_id"],
                "target": row["target_id"],
                "ships": row["ships"],
                "eta": row["chosen_eta"],
                "true_eta": row["true_eta"],
                "flags": row["risk_flags"],
            }
        )


if __name__ == "__main__":
    main()
