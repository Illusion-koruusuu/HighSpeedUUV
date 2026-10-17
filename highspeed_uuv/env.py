"""HoloOcean Dijkstra-v5 环境的任务、观测与奖励逻辑。"""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dijkstra import build_cost_field, interpolate_cost
from .sensors import RangeSensors


class DijkstraNavigationEnv(gym.Env):
    """每回合随机生成三维目标和静态球形障碍物。"""

    metadata = {"render_modes": []}

    def __init__(self, dt=0.02, max_steps=2_500, obstacle_count=2,
                 goal_guidance=False,
                 dijkstra_resolution=2.0, vertical_cost_scale=1.0):
        super().__init__()
        self.dt = float(dt)
        self.state = SimpleNamespace(
            eta=np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
            nu=np.zeros(6, dtype=float),
        )
        self._previous_action = np.zeros(3, dtype=float)
        self.ranges = RangeSensors()
        self.max_steps = int(max_steps)
        self.obstacle_count = int(obstacle_count)
        if self.obstacle_count < 0:
            raise ValueError("obstacle_count must be nonnegative")
        self.goal_guidance = bool(goal_guidance)
        self.reward_mode = "dijkstra"
        # At HoloOcean's 50 Hz step, the geodesic potential changes by only a
        # few hundredths of a metre after actuator lag.  Scale it before PPO
        # sees it so the navigation signal is not drowned by the terminal
        # reward and value-function noise.
        self.dijkstra_reward_scale = 10.0
        self.dijkstra_reward_version = 5
        self.dijkstra_depth_weight = 15.0
        self.dijkstra_alignment_weight = 5.0
        self.dijkstra_time_penalty = 0.02
        self.dijkstra_speed_shortfall_weight = 0.03
        self.dijkstra_distance_progress_weight = 2.0
        self.out_of_bounds_penalty = 1000.0
        self.dijkstra_resolution = float(dijkstra_resolution)
        if self.dijkstra_resolution <= 0.0:
            raise ValueError("dijkstra_resolution must be positive")
        self.vertical_cost_scale = float(vertical_cost_scale)
        if self.vertical_cost_scale <= 0.0:
            raise ValueError("vertical_cost_scale must be positive")
        self.planning_bounds = (
            np.array([-20.0, -70.0, 0.0]),
            np.array([100.0, 70.0, 40.0]),
        )
        self.desired_surge = 2.0
        self.goal_distance_scale = 70.0
        self.slowdown_distance = 12.0
        # The vehicle has nonzero speed and actuator lag.  Treat the goal as a
        # capture region rather than requiring an exact point crossing.
        self.goal_radius = 5.0
        # PPO is still responsible for obstacle avoidance.  When the sonar
        # is clear, these low-level terms keep actuator commands pointed at
        # the waypoint so a temporary avoidance turn does not become an orbit.
        self.goal_guidance_gain = 0.8
        self.goal_depth_guidance_gain = 1.2
        self.goal_guidance_angle = 0.7
        self.steps = 0
        self.goal = np.zeros(3)
        self.obstacles = []
        self._cost_field = None
        self._cost_origin = None
        self._cost_resolution = None

        # 动作仍为 [方向舵、水平舵、螺旋桨] 的归一化命令。
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # 任务开关、目标相对状态、左右声呐命中束数、速度。
        self.observation_space = spaces.Dict({
            "purpose": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "target": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "obstacle": spaces.Box(0.0, 61.0, shape=(2,), dtype=np.float32),
            "velocity": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        })

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _target_surge(self, distance):
        """根据剩余距离生成航速参考值，终点前逐步减速。"""
        return self.desired_surge * min(distance / self.slowdown_distance, 1.0)

    @staticmethod
    def _target_alignment(target):
        """Bounded line-of-sight alignment potential in [-1, 1].

        The previous reward used ``-abs(target)``.  That is an absolute
        observation penalty, so increasing its weight can dominate true
        distance progress and reward turning or slowing down without moving
        toward the waypoint.  Alignment excludes speed error; speed is
        handled by the arrival-speed term.
        """
        pitch, yaw = float(target[0]), float(target[1])
        return float(np.cos(pitch) * np.cos(yaw))

    @staticmethod
    def _segment_distance(point, segment_start, segment_end):
        """Return the closest distance from a point to one motion segment."""
        segment = segment_end - segment_start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-12:
            return float(np.linalg.norm(point - segment_start))
        fraction = np.clip(
            np.dot(point - segment_start, segment) / length_squared, 0.0, 1.0
        )
        closest = segment_start + fraction * segment
        return float(np.linalg.norm(point - closest))

    def _sample_scene(self):
        """在前方扇区随机放置目标和静态障碍物，避免固定地图过拟合。"""
        # 每回合随机化场景，避免策略只记住固定地图。
        goal_distance = self.np_random.uniform(50.0, 70.0)
        goal_yaw = self.np_random.uniform(-np.pi / 4, np.pi / 4)
        goal_pitch = self.np_random.uniform(-np.pi / 12, np.pi / 12)
        self.goal = np.array([
            goal_distance * np.cos(goal_pitch) * np.cos(goal_yaw),
            goal_distance * np.cos(goal_pitch) * np.sin(goal_yaw),
            np.clip(10.0 - goal_distance * np.sin(goal_pitch), 3.0, 25.0),
        ])

        self.desired_surge = float(self.np_random.uniform(1.5, 2.5))
        self.obstacles = []
        for _ in range(self.obstacle_count):
            distance = self.np_random.uniform(16.0, min(42.0, goal_distance - 8.0))
            yaw = goal_yaw + self.np_random.uniform(-np.pi / 5, np.pi / 5)
            pitch = goal_pitch + self.np_random.uniform(-np.pi / 12, np.pi / 12)
            center = np.array([
                distance * np.cos(pitch) * np.cos(yaw),
                distance * np.cos(pitch) * np.sin(yaw),
                np.clip(10.0 - distance * np.sin(pitch), 3.0, 25.0),
            ])
            radius = float(self.np_random.uniform(1.5, 3.0))
            self.obstacles.append((center, radius))

    def _build_dijkstra_field(self):
        """构造只用于奖励塑形的三维最短路径代价场。"""
        (self._cost_field, self._cost_origin, self._cost_resolution,
         self._cost_occupancy) = build_cost_field(
            self.goal,
            self.obstacles,
            start=self.state.eta[:3],
            resolution=self.dijkstra_resolution,
            obstacle_margin=1.5,
            goal_radius=self.goal_radius,
            planning_bounds=self.planning_bounds,
            vertical_cost_scale=self.vertical_cost_scale,
        )

    def _outside_planning_bounds(self, position):
        position = np.asarray(position, dtype=float)
        lower, upper = self.planning_bounds
        return bool(np.any(position < lower) or np.any(position > upper))

    def _dijkstra_cost(self, position):
        if self._cost_field is None:
            return float("inf")
        cost = interpolate_cost(
            self._cost_field, self._cost_origin, self._cost_resolution, position
        )
        if np.isfinite(cost):
            return cost

        # Leaving the finite planning volume must not turn the potential into
        # ``inf`` forever: inf -> inf would produce zero reward and allow the
        # policy to escape the map.  Assign an increasing finite extension
        # outside the grid so every additional metre away remains costly.
        finite_costs = self._cost_field[np.isfinite(self._cost_field)]
        max_cost = float(finite_costs.max()) if finite_costs.size else 0.0
        position = np.asarray(position, dtype=float)
        upper = self._cost_origin + (
            np.asarray(self._cost_field.shape, dtype=float) - 1.0
        ) * self._cost_resolution
        outside = np.maximum(self._cost_origin - position, 0.0)
        outside += np.maximum(position - upper, 0.0)
        return max_cost + float(np.linalg.norm(outside)) + float(
            np.linalg.norm(position - self.goal)
        ) * 0.05

    def _outside_dijkstra_field(self, position):
        return self._outside_planning_bounds(position)

    def _obstacle_clearance(self, position):
        if not self.obstacles:
            return float("inf")
        return min(float(np.linalg.norm(position - center) - radius)
                   for center, radius in self.obstacles)

    def _dijkstra_auxiliary_rewards(self, previous_position, position,
                                    previous_observation, observation, surge):
        """补充距离、深度和视线改善信号，帮助策略学习连续控制。"""
        old_distance = np.linalg.norm(self.goal - previous_position)
        new_distance = np.linalg.norm(self.goal - position)
        distance_progress = float(np.clip(old_distance - new_distance,
                                          -0.1, 0.1))
        old_depth_error = abs(self.goal[2] - previous_position[2])
        new_depth_error = abs(self.goal[2] - position[2])
        depth_progress = float(np.clip(old_depth_error - new_depth_error,
                                       -0.1, 0.1))
        alignment_progress = (self._target_alignment(observation["target"])
                              - self._target_alignment(previous_observation["target"]))
        # Reward useful cruise speed only in clear water and when roughly
        # facing the goal; near an obstacle or while turning, slowing is valid.
        yaw_error = float(observation["target"][1])
        speed_gate = (1.0 - float(observation["purpose"][0])) * max(
            0.0, np.cos(yaw_error)
        )
        speed_shortfall = speed_gate * max(
            0.0, self._target_surge(np.linalg.norm(self.goal - position)) - surge
        )
        return (distance_progress, depth_progress, float(alignment_progress),
                float(speed_shortfall))

    def observation_from_state(self, eta, nu):
        """由任意 NED 位姿和速度构造策略观测，供 HoloOcean 回放复用。"""
        front_range = self.ranges._scan(eta, self.ranges.front_angles,
                                        self.obstacles)                             # 扫描前方声呐数据
        detected = front_range < self.ranges.max_range                              # 是否有障碍物命中
        left_beams = np.count_nonzero(detected & (self.ranges.front_angles > 0))    # 左右声呐命中束数
        right_beams = np.count_nonzero(detected & (self.ranges.front_angles < 0))
        avoiding = float(left_beams + right_beams > 0)                               # 是否处于避碰状态

        delta = self.goal - eta[:3]
        target_distance = float(np.linalg.norm(delta))
        horizontal_distance = np.hypot(delta[0], delta[1])
        target_pitch = np.arctan2(-delta[2], horizontal_distance) - eta[4]
        target_yaw = self._wrap_angle(np.arctan2(delta[1], delta[0]) - eta[5])
        # 加入归一化距离，使策略能区分远距离巡航和接近目标后的精细对准。
        normalized_distance = np.clip(target_distance / self.goal_distance_scale,
                                      0.0, 1.0)
        # 推进器不能反转制动，因此在终点前预留减速距离，避免高速擦过目标。
        target_surge = self._target_surge(target_distance)
        # 保留目标信息；purpose 单独表明正在避碰，避免策略转向后丢失
        # 重新对准目标的依据。这是相对论文“完全屏蔽目标”的工程调整。
        target = np.array([
            target_pitch,
            target_yaw,
            target_surge - nu[0],
            normalized_distance,
        ])
        return {
            "purpose": np.array([avoiding], dtype=np.float32),
            "target": target.astype(np.float32),
            "obstacle": np.array([right_beams, left_beams], dtype=np.float32),
            "velocity": nu[[0, 1, 2, 4, 5]].astype(np.float32),
        }

    def _observe(self):
        return self.observation_from_state(self.state.eta, self.state.nu)

    def _apply_goal_guidance(self, action, observation=None):
        """Bias clear-water commands toward the current goal direction."""
        if not self.goal_guidance:
            return action
        if observation is None:
            observation = self._observe()
        if observation["purpose"][0] > 0.0:
            return action
        scale = self.goal_guidance_gain
        angle = self.goal_guidance_angle
        yaw_error = float(observation["target"][1])
        pitch_error = float(observation["target"][0])
        normalized_distance = float(observation["target"][3])
        # Combine the learned stern-plane command with a bounded depth bias;
        # the policy remains in control and can still choose a different
        # three-dimensional path when that improves its return.
        action = action.copy()
        action[0] += scale * np.tanh(yaw_error / angle)
        action[1] -= self.goal_depth_guidance_gain * np.tanh(pitch_error / angle)
        # Slow down before the turn rather than after passing the waypoint.
        # The cosine term turns a large bearing error into an early throttle
        # reduction, while the floor prevents the vehicle from stopping
        # completely before it has entered the capture region.
        bearing_factor = max(0.25, np.cos(yaw_error))
        distance = normalized_distance * self.goal_distance_scale
        target_speed = self._target_surge(distance) * bearing_factor
        speed_ratio = np.clip(target_speed / max(self.desired_surge, 1e-6), 0.0, 1.0)
        throttle_limit = 2.0 * speed_ratio - 1.0
        action[2] = min(action[2], throttle_limit)
        return np.clip(action, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state.eta = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        self.state.nu = np.zeros(6, dtype=float)
        self._sample_scene()
        self._build_dijkstra_field()
        self.steps = 0
        return self._observe(), {}

    def step(self, action):
        raise NotImplementedError("动力学由 HoloOcean Torpedo 环境实现")

    def close(self):
        """Gymnasium 兼容的空关闭钩子；Torpedo 本身无外部进程。"""
