"""三路声呐空间方向、观测契约和 CNN 特征提取器测试。"""
import unittest

import numpy as np
import torch
from gymnasium import spaces

from highspeed_uuv.policy import BoundedActorCriticPolicy, SonarCNNFeaturesExtractor
from highspeed_uuv.sensors import SonarArray


class SonarObservationTests(unittest.TestCase):
    def setUp(self):
        self.sonar = SonarArray(elevation_bins=24, azimuth_bins=64)
        self.eta = np.zeros(6, dtype=float)

    def test_three_view_maps_detect_forward_and_both_sides(self):
        obstacles = [
            (np.array([12.0, 0.0, 0.0]), 2.0),
            (np.array([0.0, 12.0, 0.0]), 2.0),
            (np.array([0.0, -12.0, 0.0]), 2.0),
        ]
        observation = self.sonar.observe(self.eta, obstacles)
        self.assertEqual(observation.shape, (6, 24, 64))
        self.assertTrue(np.isfinite(observation).all())
        self.assertEqual(self.sonar.hit_count(observation) > 0, True)
        for hit_channel in (1, 3, 5):
            self.assertGreater(float(observation[hit_channel].sum()), 0.0)

    def test_empty_scene_is_max_range_without_echo(self):
        observation = self.sonar.observe(self.eta, [])
        self.assertTrue(np.all(observation[[0, 2, 4]] == 1.0))
        self.assertFalse(np.any(observation[[1, 3, 5]]))

    def test_ned_left_and_right_are_not_swapped(self):
        left = self.sonar.observe(
            self.eta, [(np.array([0.0, -12.0, 0.0]), 2.0)]
        )
        right = self.sonar.observe(
            self.eta, [(np.array([0.0, 12.0, 0.0]), 2.0)]
        )
        self.assertGreater(left[3].sum(), 0.0)
        self.assertEqual(left[5].sum(), 0.0)
        self.assertGreater(right[5].sum(), 0.0)
        self.assertEqual(right[3].sum(), 0.0)

    def test_far_echo_does_not_disable_cruise(self):
        far = self.sonar.observe(
            self.eta, [(np.array([30.0, 0.0, 0.0]), 2.0)]
        )
        near = self.sonar.observe(
            self.eta, [(np.array([12.0, 0.0, 0.0]), 2.0)]
        )
        self.assertGreater(self.sonar.hit_count(far), 0)
        self.assertFalse(self.sonar.near_obstacle(far))
        self.assertTrue(self.sonar.near_obstacle(near))

    def test_cnn_extractor_accepts_dict_observation(self):
        observation_space = spaces.Dict({
            "sonar": spaces.Box(0.0, 1.0, shape=(6, 24, 64), dtype=np.float32),
            "target": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "path": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "velocity": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        })
        extractor = SonarCNNFeaturesExtractor(observation_space, features_dim=256)
        output = extractor({
            "sonar": torch.zeros(2, 6, 24, 64),
            "target": torch.zeros(2, 4),
            "path": torch.zeros(2, 4),
            "velocity": torch.zeros(2, 5),
        })
        self.assertEqual(tuple(output.shape), (2, 256))

    def test_environment_can_include_dijkstra_path_observation(self):
        from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv

        env = HoloOceanTorpedoEnv(
            max_steps=4, obstacle_count=0, include_path_observation=True,
            easy_rehearsal_probability=1.0,
        )
        try:
            observation, _ = env.reset(seed=0)
            self.assertIn("path", observation)
            self.assertEqual(observation["path"].shape, (4,))
            self.assertTrue(np.isfinite(observation["path"]).all())
            self.assertEqual(len(env.obstacles), 0)
            self.assertLessEqual(np.linalg.norm(env.goal - env.state.eta[:3]), 40.0)
        finally:
            env.close()

    def test_bounded_policy_uses_squashed_gaussian(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
        from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv

        env = HoloOceanTorpedoEnv(
            max_steps=4, obstacle_count=0, goal_guidance=False,
            capture_assist=False, cruise_assist=False,
        )
        try:
            model = PPO(
                BoundedActorCriticPolicy, env, n_steps=8, batch_size=8,
                policy_kwargs={"features_extractor_class": SonarCNNFeaturesExtractor},
            )
            self.assertIsInstance(
                model.policy.action_dist, SquashedDiagGaussianDistribution
            )
            observation, _ = env.reset(seed=0)
            action, _ = model.predict(observation, deterministic=False)
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <= 1.0))
        finally:
            env.close()

    def test_curriculum_has_four_progressive_stages(self):
        from training.train import GoalObstacleCurriculumCallback

        callback = GoalObstacleCurriculumCallback(
            2, 5, (50.0, 70.0), np.deg2rad(45.0), np.deg2rad(15.0),
            1_000_000,
        )
        self.assertEqual(callback._stage(0.05)[4], 1)
        self.assertEqual(callback._stage(0.20)[4], 2)
        self.assertEqual(callback._stage(0.45)[4], 3)
        self.assertEqual(callback._stage(0.80)[4], 4)

    def test_new_reward_version_reduces_terminal_scale(self):
        from highspeed_uuv.env import DijkstraNavigationEnv

        env = DijkstraNavigationEnv(obstacle_count=0)
        self.assertEqual(env.dijkstra_reward_version, 12)
        self.assertEqual(env.success_reward, 150.0)
        self.assertEqual(env.collision_penalty, 300.0)
        self.assertEqual(env.out_of_bounds_penalty, 300.0)
        self.assertGreater(env.dijkstra_distance_progress_weight, 2.0)
        self.assertGreater(env.dijkstra_speed_shortfall_weight, 0.1)
        self.assertGreater(env.dijkstra_alignment_error_weight, 0.0)
        env.close()

    def test_capture_reward_prefers_radial_approach_over_tangential_motion(self):
        from highspeed_uuv.env import DijkstraNavigationEnv

        env = DijkstraNavigationEnv(obstacle_count=0)
        env.reset(seed=0)
        env.goal = np.array([10.0, 0.0, 10.0])
        env.state.eta[:3] = np.array([8.0, 0.0, 10.0])
        env.state.nu[:] = 0.0
        observation = env._observe()
        approach = env._capture_navigation_rewards(
            np.array([7.0, 0.0, 10.0]), env.state.eta[:3], observation, 1.0
        )[0]
        tangent = env._capture_navigation_rewards(
            np.array([8.0, 0.0, 10.0]), env.state.eta[:3], observation, 1.0
        )[0]
        self.assertGreater(approach, tangent)
        env.close()

    def test_cruise_floor_keeps_minimum_forward_command(self):
        from highspeed_uuv.env import DijkstraNavigationEnv

        env = DijkstraNavigationEnv(obstacle_count=0)
        observation = {
            "sonar": np.zeros(env.sonar.observation_shape, dtype=np.float32),
            "target": np.array([0.0, 0.0, 2.0, 0.8], dtype=np.float32),
            "velocity": np.zeros(5, dtype=np.float32),
        }
        action = env._apply_cruise_floor(np.array([0.0, 0.0, -1.0]), observation)
        self.assertEqual(action[2], env.cruise_min_throttle)
        threat = observation.copy()
        threat["sonar"] = self.sonar.observe(
            self.eta, [(np.array([12.0, 0.0, 0.0]), 2.0)]
        )
        action = env._apply_cruise_floor(np.array([0.0, 0.0, -1.0]), threat)
        self.assertEqual(action[2], env.avoidance_min_throttle)
        observation["velocity"][0] = 1.0
        action = env._apply_cruise_floor(np.array([0.0, 0.0, -1.0]), observation)
        self.assertEqual(action[2], env.cruise_min_throttle)
        observation["velocity"][0] = env.cruise_floor_release_speed
        action = env._apply_cruise_floor(np.array([0.0, 0.0, -1.0]), observation)
        self.assertEqual(action[2], -1.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
