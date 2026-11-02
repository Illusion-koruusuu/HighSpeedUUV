"""目标捕获残差的方向和油门约束测试。"""
import unittest

import numpy as np

from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv


class CaptureControlTests(unittest.TestCase):
    def setUp(self):
        self.env = HoloOceanTorpedoEnv(max_steps=2, obstacle_count=0,
                                       vehicle_profile="boosted")

    def tearDown(self):
        self.env.close()

    def test_residual_turns_toward_negative_bearing(self):
        observation = {
            "sonar": np.zeros(self.env.sonar.observation_shape, dtype=np.float32),
            "target": np.array([-0.5, -1.0, 0.0, 0.1], dtype=np.float32),
        }
        action = self.env._keep_turning_near_goal(
            np.array([0.8, 0.0, -1.0]), observation
        )
        self.assertLess(action[0], 0.0)
        self.assertGreaterEqual(action[2], self.env.capture_min_throttle)

    def test_obstacle_mode_does_not_modify_policy_action(self):
        observation = {
            "sonar": np.zeros(self.env.sonar.observation_shape, dtype=np.float32),
            "target": np.array([0.5, -1.0, 0.0, 0.1], dtype=np.float32),
        }
        observation["sonar"][1, 12, 32] = 1.0
        action = np.array([0.8, -0.2, -1.0])
        np.testing.assert_allclose(
            self.env._keep_turning_near_goal(action, observation), action
        )


if __name__ == "__main__":
    unittest.main()
