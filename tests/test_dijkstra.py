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


if __name__ == "__main__":
    unittest.main()
