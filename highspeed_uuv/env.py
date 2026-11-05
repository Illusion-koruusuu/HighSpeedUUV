"""HoloOcean Dijkstra-v5 环境的任务、观测与奖励逻辑。"""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dijkstra import build_cost_field, interpolate_cost
from .sensors import SonarArray


class DijkstraNavigationEnv(gym.Env):
    """每回合随机生成三维目标和静态球形障碍物。"""

    metadata = {"render_modes": []}

    def __init__(self, dt=0.02, max_steps=2_500, obstacle_count=None,
                 goal_guidance=False, cruise_assist=True,
                 include_path_observation=False,
                 easy_rehearsal_probability=0.0,
                 dijkstra_resolution=2.0, vertical_cost_scale=1.0,
                 obstacle_count_range=(2, 5)):
        super().__init__()
        self.dt = float(dt)
        self.state = SimpleNamespace(
            eta=np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
            nu=np.zeros(6, dtype=float),
        )
        self._previous_action = np.zeros(3, dtype=float)
        self.sonar = SonarArray()
        self.max_steps = int(max_steps)
        if obstacle_count is not None:
            obstacle_count_range = (int(obstacle_count), int(obstacle_count))
        if len(obstacle_count_range) != 2:
            raise ValueError("obstacle_count_range must contain two integers")
        self.obstacle_count_min, self.obstacle_count_max = map(
            int, obstacle_count_range
        )
        if (self.obstacle_count_min < 0
                or self.obstacle_count_max < self.obstacle_count_min):
            raise ValueError("invalid obstacle_count_range")
        # 每回合在范围内随机采样，障碍物之间不做排斥检查，因此允许重叠。
        self.obstacle_count = self.obstacle_count_min
        self.goal_guidance = bool(goal_guidance)
        self.cruise_assist = bool(cruise_assist)
        self.include_path_observation = bool(include_path_observation)
        self.easy_rehearsal_probability = float(easy_rehearsal_probability)
        if not 0.0 <= self.easy_rehearsal_probability <= 1.0:
            raise ValueError("easy_rehearsal_probability must be in [0, 1]")
        self.easy_rehearsal_goal_range = (25.0, 40.0)
        self.goal_distance_range = (50.0, 70.0)
        self.goal_yaw_limit = np.pi / 4.0
        self.goal_pitch_limit = np.pi / 12.0
        self.reward_mode = "dijkstra"
        # At HoloOcean's 50 Hz step, the geodesic potential changes by only a
        # few hundredths of a metre after actuator lag.  Scale it before PPO
        # sees it so the navigation signal is not drowned by the terminal
        # reward and value-function noise.
        self.dijkstra_reward_scale = 10.0
        self.dijkstra_reward_version = 12
        self.dijkstra_depth_weight = 15.0
        self.dijkstra_alignment_weight = 5.0
        self.dijkstra_alignment_error_weight = 0.50
        # 终止项和逐步塑形奖励保持相近量级，避免 critic 只拟合成败标签。
        self.success_reward = 150.0
        self.collision_penalty = 300.0
        self.out_of_bounds_penalty = 300.0
        self.dijkstra_time_penalty = 0.05
        self.dijkstra_speed_shortfall_weight = 0.50
        self.dijkstra_distance_progress_weight = 8.0
        self.dijkstra_cruise_speed_weight = 0.50
        self.dijkstra_path_follow_weight = 4.0
        # 目标捕获阶段：绝对方向误差、径向接近和目标方向速度。
        self.capture_reward_distance = 20.0
        self.capture_alignment_weight = 3.0
        self.capture_radial_progress_weight = 15.0
        self.capture_radial_velocity_weight = 1.5
        # 清水巡航时不能长期把油门压到怠速，否则舵面没有转向力且距离进展为零。
        # 归一化油门 -0.2 对应约 1200 rpm，在当前 Fossen 参数下可维持约 2 m/s。
        self.cruise_min_throttle = -0.20
        # 近障碍时允许明显降速，但不能降到执行器失去水动力的怠速区。
        self.avoidance_min_throttle = -0.50
        self.cruise_min_distance = 8.0
        self.cruise_floor_enter_speed = 1.2
        self.cruise_floor_release_speed = 1.5
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
        # 50 Hz 积分与执行器滞后会让单步轨迹跨过边界；6 m 捕获球给出
        # 与当前高速模型相称的实际到达判据，同时 Dijkstra 目标区域同步扩大。
        self.goal_radius = 6.0
        # 舵面只有在有前进速度时才有足够的转向力。目标附近若完全收油，
        # 潜航器会停在捕获球外，之后无法重新对准目标。
        # 留出六个捕获半径的提前量，弥补潜航器转弯惯性和舵面滞后。
        self.capture_turn_distance = 6.0 * self.goal_radius
        self.capture_turn_angle = 0.20
        self.capture_min_throttle = -0.90
        self.capture_steering_gain = 1.2
        self.capture_vertical_gain = 1.8
        self.capture_min_rudder = 0.65
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
        # 声呐输入保留三路二维回波图，由 CNN 学习空间障碍特征。
        observation_spaces = {
            "sonar": spaces.Box(0.0, 1.0, shape=self.sonar.observation_shape,
                                dtype=np.float32),
            "target": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "velocity": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        }
        if self.include_path_observation:
            observation_spaces["path"] = spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(observation_spaces)

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
        rehearsal = (self.np_random.random()
                     < self.easy_rehearsal_probability)
        if rehearsal:
            goal_distance = self.np_random.uniform(
                *self.easy_rehearsal_goal_range
            )
            goal_yaw = self.np_random.uniform(-np.deg2rad(30.0),
                                              np.deg2rad(30.0))
            goal_pitch = self.np_random.uniform(-np.deg2rad(10.0),
                                                np.deg2rad(10.0))
        else:
            goal_distance = self.np_random.uniform(*self.goal_distance_range)
            goal_yaw = self.np_random.uniform(-self.goal_yaw_limit,
                                              self.goal_yaw_limit)
            goal_pitch = self.np_random.uniform(-self.goal_pitch_limit,
                                                self.goal_pitch_limit)
        self.goal = np.array([
            goal_distance * np.cos(goal_pitch) * np.cos(goal_yaw),
            goal_distance * np.cos(goal_pitch) * np.sin(goal_yaw),
            np.clip(10.0 - goal_distance * np.sin(goal_pitch), 3.0, 25.0),
        ])

        self.desired_surge = float(self.np_random.uniform(1.5, 2.5))
        self.obstacle_count = 0 if rehearsal else int(self.np_random.integers(
            self.obstacle_count_min, self.obstacle_count_max + 1
        ))
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
            radius = float(self.np_random.uniform(1.5, 3.5))
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

    def _plan_dijkstra_path(self, start, max_points=2000):
        """从当前起点沿已有代价场提取到目标的滚动最短路径。

        目标和静态障碍物不变时，重新搜索整张场没有新信息；改变起点后重新
        提取下降路径，得到的路径前缀才是本步对应的最短路径。
        """
        if self._cost_field is None:
            return np.empty((0, 3), dtype=float)
        shape = np.asarray(self._cost_field.shape, dtype=int)
        current = tuple(np.rint(
            (np.asarray(start, dtype=float) - self._cost_origin)
            / self._cost_resolution
        ).astype(int).clip(0, shape - 1))
        path = [self._cost_origin + np.asarray(current) * self._cost_resolution]
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
        for _ in range(max_points):
            current_cost = float(self._cost_field[current])
            if not np.isfinite(current_cost) or current_cost <= 1e-5:
                break
            candidates = []
            for dx, dy, dz in offsets:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if any(index < 0 or index >= shape[axis]
                       for axis, index in enumerate(neighbor)):
                    continue
                if self._cost_occupancy[neighbor]:
                    continue
                # 对角移动时检查所有经过的面相邻格，避免从膨胀障碍物棱角斜穿。
                axes = [axis for axis, step in enumerate((dx, dy, dz)) if step]
                blocked_corner = False
                for mask in range(1, (1 << len(axes)) - 1):
                    probe = list(current)
                    for bit, axis in enumerate(axes):
                        if mask & (1 << bit):
                            probe[axis] += (dx, dy, dz)[axis]
                    if self._cost_occupancy[tuple(probe)]:
                        blocked_corner = True
                        break
                if blocked_corner:
                    continue
                neighbor_cost = float(self._cost_field[neighbor])
                if np.isfinite(neighbor_cost):
                    candidates.append((neighbor_cost, neighbor))
            if not candidates:
                break
            next_cost, next_index = min(candidates, key=lambda item: item[0])
            if next_cost >= current_cost - 1e-6:
                break
            current = next_index
            path.append(self._cost_origin + np.asarray(current)
                        * self._cost_resolution)
        return np.asarray(path, dtype=float)

    @staticmethod
    def _path_progress_and_error(position, path):
        """返回位置在规划路径上的弧长进展和到路径的最近距离。"""
        if len(path) < 2:
            return 0.0, float("inf")
        best_error = float("inf")
        best_progress = 0.0
        accumulated = 0.0
        point = np.asarray(position, dtype=float)
        for start, end in zip(path[:-1], path[1:]):
            segment = end - start
            length = float(np.linalg.norm(segment))
            if length <= 1e-9:
                continue
            fraction = float(np.clip(
                np.dot(point - start, segment) / (length * length), 0.0, 1.0
            ))
            error = float(np.linalg.norm(point - (start + fraction * segment)))
            if error < best_error:
                best_error = error
                best_progress = accumulated + fraction * length
            accumulated += length
        return best_progress, best_error

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
        speed_gate = (0.0 if self._is_avoiding(observation) else 1.0) * max(
            0.0, np.cos(yaw_error)
        )
        speed_shortfall = speed_gate * max(
            0.0, self._target_surge(np.linalg.norm(self.goal - position)) - surge
        )
        return (distance_progress, depth_progress, float(alignment_progress),
                float(speed_shortfall))

    def _navigation_alignment_error(self, observation):
        """Penalize the current heading error, not only its change."""
        key = "path" if "path" in observation else "target"
        pitch_error = abs(float(observation[key][0]))
        yaw_error = abs(float(observation[key][1]))
        return float(np.clip(
            0.7 * yaw_error / np.pi + 0.3 * pitch_error / (np.pi / 2.0),
            0.0, 1.0,
        ))

    def _capture_navigation_rewards(self, previous_position, position,
                                    observation, surge):
        """目标附近的末端捕获奖励，专门抑制等距绕圈。"""
        distance = float(np.linalg.norm(self.goal - position))
        gate = float(np.clip(
            (self.capture_reward_distance - distance)
            / self.capture_reward_distance, 0.0, 1.0,
        ))
        if gate <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        # alignment_progress 只看角度变化；这里增加当前绝对误差的持续惩罚。
        pitch_error = abs(float(observation["target"][0]))
        yaw_error = abs(float(observation["target"][1]))
        alignment_error = float(np.clip(
            0.5 * yaw_error / np.pi
            + 0.5 * pitch_error / (np.pi / 2.0),
            0.0, 1.0,
        ))
        alignment_reward = -alignment_error

        previous_distance = float(np.linalg.norm(self.goal - previous_position))
        radial_progress = float(np.clip(
            previous_distance - distance, -0.1, 0.1
        ))

        # 体坐标线速度转换为 NED 速度，计算朝向目标的速度投影。
        world_velocity = self.sonar._rotation_matrix(self.state.eta) @ self.state.nu[:3]
        goal_delta = self.goal - position
        goal_norm = max(float(np.linalg.norm(goal_delta)), 1e-6)
        radial_velocity = float(np.dot(world_velocity, goal_delta / goal_norm))
        radial_velocity_reward = float(np.clip(
            radial_velocity / max(self._target_surge(distance), 0.5),
            -1.0, 1.0,
        ))
        _ = surge  # 保留显式参数，避免误用角速度替代线速度。
        total = gate * (
            self.capture_alignment_weight * alignment_reward
            + self.capture_radial_progress_weight * radial_progress
            + self.capture_radial_velocity_weight * radial_velocity_reward
        )
        return (float(total), float(alignment_reward),
                float(radial_progress), float(radial_velocity_reward))

    def observation_from_state(self, eta, nu, planned_path=None):
        """由任意 NED 位姿和速度构造策略观测，供 HoloOcean 回放复用。"""
        sonar = self.sonar.observe(eta, self.obstacles)

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
        # 保留目标信息，使策略绕开障碍后还能重新对准目标。
        target = np.array([
            target_pitch,
            target_yaw,
            target_surge - nu[0],
            normalized_distance,
        ])
        observation = {
            "sonar": sonar,
            "target": target.astype(np.float32),
            "velocity": nu[[0, 1, 2, 4, 5]].astype(np.float32),
        }
        if self.include_path_observation:
            observation["path"] = (
                self._path_observation_from_path(eta, planned_path)
                if planned_path is not None else self._path_observation(eta)
            )
        return observation

    def _is_avoiding(self, observation):
        """用近距离前向/舷侧回波判断是否确实需要即时避障。"""
        return self.sonar.near_obstacle(observation["sonar"])

    def _apply_cruise_floor(self, action, observation):
        """只在艇速塌到近乎停航时恢复最低机动速度，随后交还 PPO。"""
        distance = float(observation["target"][3]) * self.goal_distance_scale
        surge = float(observation["velocity"][0])
        if (distance <= self.cruise_min_distance
                or surge >= self.cruise_floor_release_speed):
            return action
        if surge >= self.cruise_floor_enter_speed:
            # 滞回带内先不切换动作，避免在阈值附近反复开关推进下限。
            return action
        action = action.copy()
        minimum = (self.avoidance_min_throttle
                   if self._is_avoiding(observation)
                   else self.cruise_min_throttle)
        action[2] = max(float(action[2]), minimum)
        return np.clip(action, -1.0, 1.0)

    def _path_observation(self, eta):
        """Return local Dijkstra path direction and remaining path cost."""
        if self._cost_field is None:
            return np.zeros(4, dtype=np.float32)
        path = self._plan_dijkstra_path(eta[:3], max_points=160)
        return self._path_observation_from_path(eta, path)

    def _path_observation_from_path(self, eta, path):
        """Build path features from a previously extracted Dijkstra prefix."""
        if len(path) < 2:
            return np.zeros(4, dtype=np.float32)
        point = np.asarray(eta[:3], dtype=float)
        distances = np.linalg.norm(path - point, axis=1)
        start = int(np.argmin(distances))
        lookahead = path[-1]
        for candidate in path[start:]:
            if np.linalg.norm(candidate - point) >= 8.0:
                lookahead = candidate
                break
        delta = lookahead - point
        horizontal = float(np.hypot(delta[0], delta[1]))
        target_pitch = float(np.arctan2(-delta[2], max(horizontal, 1e-6))
                             - eta[4])
        target_yaw = float(self._wrap_angle(
            np.arctan2(delta[1], delta[0]) - eta[5]
        ))
        lookahead_distance = float(np.linalg.norm(delta))
        remaining_cost = self._dijkstra_cost(point)
        remaining_cost_norm = (float(np.clip(remaining_cost / 70.0, 0.0, 2.0))
                               if np.isfinite(remaining_cost) else 2.0)
        return np.array([
            target_pitch,
            target_yaw,
            lookahead_distance / 10.0,
            remaining_cost_norm,
        ], dtype=np.float32)

    def _observe(self, planned_path=None):
        return self.observation_from_state(
            self.state.eta, self.state.nu, planned_path=planned_path
        )

    def _apply_goal_guidance(self, action, observation=None):
        """Bias clear-water commands toward the current goal direction."""
        if not self.goal_guidance:
            return action
        if observation is None:
            observation = self._observe()
        if self._is_avoiding(observation):
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

    def _keep_turning_near_goal(self, action, observation):
        """目标附近做有限残差修正，避免低速时错过目标后继续绕行。"""
        distance = float(observation["target"][3]) * self.goal_distance_scale
        pitch_error = abs(float(observation["target"][0]))
        yaw_error = abs(float(observation["target"][1]))
        if (distance <= self.capture_turn_distance
                and not self._is_avoiding(observation)
                and max(pitch_error, yaw_error) > self.capture_turn_angle):
            action = action.copy()
            # 这是残差而非直接覆盖：保留一半 PPO 舵面输出，再加上有界
            # 的目标方向修正；因此远距离规划和障碍绕行仍由策略决定。
            gain = self.capture_steering_gain
            action[0] = 0.5 * action[0] + gain * np.tanh(
                float(observation["target"][1]) / 0.7
            )
            # 末端若 PPO 仍给出相反舵向，至少纠正舵向；不强行指定完整舵角。
            yaw = float(observation["target"][1])
            if abs(yaw) > self.capture_turn_angle:
                action[0] = np.sign(yaw) * max(abs(float(action[0])),
                                                self.capture_min_rudder)
            action[1] = 0.5 * action[1] - self.capture_vertical_gain * np.tanh(
                float(observation["target"][0]) / 0.7
            )
            action[2] = max(float(action[2]), self.capture_min_throttle)
        return np.clip(action, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state.eta = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        self.state.nu = np.zeros(6, dtype=float)
        self._sample_scene()
        self._build_dijkstra_field()
        self.steps = 0
        self._previous_action.fill(0.0)
        return self._observe(), {}

    def step(self, action):
        raise NotImplementedError("动力学由 HoloOcean Torpedo 环境实现")

    def close(self):
        """Gymnasium 兼容的空关闭钩子；Torpedo 本身无外部进程。"""
