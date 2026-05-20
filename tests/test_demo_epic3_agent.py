from __future__ import annotations

import unittest

from demo_epic3_agent import agent


class DemoEpic3AgentTests(unittest.TestCase):
    def test_agent_runs_on_minimal_observation_without_custom_fleet_helpers(self) -> None:
        observation = {
            "step": 0,
            "player_id": 0,
            "planets": [
                [0, 0, 10.0, 10.0, 1.0, 10, 1],
                [1, -1, 20.0, 10.0, 1.0, 3, 2],
                [2, 1, 90.0, 90.0, 1.0, 10, 1],
            ],
            "fleets": [],
            "remainingOverageTime": 60,
        }
        config = {
            "episodeSteps": 500,
            "shipSpeed": 6.0,
            "boardSize": 100,
            "sunRadius": 10.0,
            "sunX": 50.0,
            "sunY": 50.0,
        }

        action = agent(observation, config)

        self.assertIsInstance(action, list)


if __name__ == "__main__":
    unittest.main()
