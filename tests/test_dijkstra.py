"""Dijkstra 奖励场的基础性质测试。"""
import unittest

import numpy as np

from highspeed_uuv.dijkstra import build_cost_field, interpolate_cost


class DijkstraFieldTests(unittest.TestCase):
    def test_fixed_bounds_and_goal_have_zero_cost(self):
        bounds = (np.array([-20, -70, 0]), np.array([100, 70, 40]))
        goal = np.array([50.0, 10.0, 12.0])
        field, origin, resolution, _ = build_cost_field(
            goal, [], resolution=2.0, goal_radius=5.0,
            planning_bounds=bounds, vertical_cost_scale=3.0,
        )
        self.assertTrue(np.allclose(origin, bounds[0]))
        self.assertEqual(field.shape, (61, 71, 21))
        self.assertEqual(interpolate_cost(field, origin, resolution, goal), 0.0)

    def test_obstacle_inflation_blocks_cells(self):
        goal = np.array([20.0, 0.0, 10.0])
        obstacle = (np.array([10.0, 0.0, 10.0]), 2.0)
        bounds = (np.array([-2, -8, 2]), np.array([24, 8, 18]))
        field, origin, resolution, occupancy = build_cost_field(
            goal, [obstacle], resolution=2.0, obstacle_margin=1.5,
            goal_radius=2.0, planning_bounds=bounds,
        )
        blocked = np.rint((obstacle[0] - origin) / resolution).astype(int)
        self.assertTrue(occupancy[tuple(blocked)])
        self.assertTrue(np.isfinite(interpolate_cost(field, origin, resolution, [0, 0, 10])))

    def test_rolling_path_starts_at_current_state_and_reaches_goal_region(self):
        from highspeed_uuv.env import DijkstraNavigationEnv

        env = DijkstraNavigationEnv(obstacle_count=0)
        env.reset(seed=4)
        env.goal = np.array([30.0, 8.0, 12.0])
        env._build_dijkstra_field()
        start = np.array([8.0, -6.0, 16.0])
        path = env._plan_dijkstra_path(start)
        np.testing.assert_allclose(path[0], env._cost_origin + np.rint(
            (start - env._cost_origin) / env._cost_resolution
        ) * env._cost_resolution)
        self.assertGreater(len(path), 1)
        self.assertLessEqual(np.linalg.norm(path[-1] - env.goal), env.goal_radius + 2.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
