from __future__ import annotations

import json
import unittest

from core.types import Fleet, GameState, Planet
from rl.rewards import (
    RewardConfig,
    RewardContext,
    calculate_reward,
    reward_config_metadata,
)
from rl.types import Candidate, CandidateType


def _state(
    *,
    step: int = 0,
    player_id: int = 0,
    planets: list[Planet],
    fleets: list[Fleet] | None = None,
) -> GameState:
    return GameState(
        step=step,
        player_id=player_id,
        planets=planets,
        fleets=list(fleets or []),
        config={"episodeSteps": 500},
    )


class RewardCalculationTests(unittest.TestCase):
    def test_terminal_win_loss_and_draw_are_primary(self) -> None:
        draw_state = _state(
            planets=[
                Planet(id=0, owner=0, ships=20, production=1),
                Planet(id=1, owner=1, ships=20, production=1),
            ],
        )
        win_state = _state(
            planets=[
                Planet(id=0, owner=0, ships=30, production=1),
                Planet(id=1, owner=1, ships=10, production=1),
            ],
        )
        loss_state = _state(
            planets=[
                Planet(id=0, owner=0, ships=10, production=1),
                Planet(id=1, owner=1, ships=30, production=1),
            ],
        )

        draw_breakdown = calculate_reward(RewardContext(None, draw_state, done=True))
        win_breakdown = calculate_reward(RewardContext(None, win_state, done=True))
        loss_breakdown = calculate_reward(RewardContext(None, loss_state, done=True))

        self.assertEqual(draw_breakdown.terminal, 0.0)
        self.assertEqual(draw_breakdown.total, 0.0)
        self.assertEqual(win_breakdown.terminal, 1.0)
        self.assertGreaterEqual(win_breakdown.total, 1.0)
        self.assertEqual(loss_breakdown.terminal, -1.0)
        self.assertLessEqual(loss_breakdown.total, -1.0)

    def test_shaping_terms_are_configurable_and_clipped_lightly(self) -> None:
        previous = _state(
            planets=[
                Planet(id=0, owner=0, ships=40, production=2),
                Planet(id=1, owner=1, ships=10, production=1),
                Planet(id=2, owner=-1, ships=4, production=3),
            ],
        )
        current = _state(
            step=1,
            planets=[
                Planet(id=0, owner=0, ships=35, production=2),
                Planet(id=1, owner=0, ships=9, production=1),
                Planet(id=2, owner=0, ships=1, production=3),
            ],
        )
        candidate = Candidate(
            CandidateType.ATTACK,
            source_id=0,
            target_id=2,
            ships=12,
            angle=0.0,
            eta=5,
        )
        config = RewardConfig(
            production_swing_weight=0.1,
            planet_capture_weight=0.3,
            planet_loss_weight=0.2,
            overkill_weight=0.05,
            shaping_clip=0.25,
        )

        breakdown = calculate_reward(
            RewardContext(previous, current, done=False, submitted_candidates=(candidate,)),
            config,
        )

        self.assertEqual(breakdown.terminal, 0.0)
        self.assertGreater(breakdown.production_swing, 0.0)
        self.assertGreater(breakdown.planets_captured, 0.0)
        self.assertEqual(breakdown.planets_lost, 0.0)
        self.assertLess(breakdown.overkill_waste, 0.0)
        self.assertLessEqual(abs(breakdown.shaping), config.shaping_clip)
        self.assertEqual(breakdown.total, breakdown.shaping)

    def test_reward_ignores_legacy_heuristic_agreement_signal(self) -> None:
        previous = _state(
            planets=[
                Planet(id=0, owner=0, ships=20, production=2),
                Planet(id=1, owner=1, ships=20, production=2),
            ],
        )
        current = _state(
            step=1,
            planets=[
                Planet(id=0, owner=0, ships=20, production=2),
                Planet(id=1, owner=1, ships=20, production=2),
            ],
        )

        without_signal = calculate_reward(RewardContext(previous, current, done=False))
        with_signal = calculate_reward(
            RewardContext(
                previous,
                current,
                done=False,
                trainer_info={"heuristic_agreement": 1.0},
            )
        )

        self.assertEqual(without_signal.metadata(), with_signal.metadata())
        metadata = reward_config_metadata(RewardConfig())
        self.assertFalse(any("heuristic" in key for key in metadata))

    def test_reward_config_metadata_is_stable_and_serializable(self) -> None:
        config = RewardConfig(
            production_swing_weight=0.03,
            planet_capture_weight=0.07,
            sun_waste_weight=0.01,
        )

        metadata = reward_config_metadata(config)
        encoded = json.dumps(metadata, sort_keys=True)

        self.assertIn("schema_version", metadata)
        self.assertIn("\"production_swing_weight\": 0.03", encoded)
        self.assertEqual(metadata["sun_waste_weight"], 0.01)


if __name__ == "__main__":
    unittest.main()
