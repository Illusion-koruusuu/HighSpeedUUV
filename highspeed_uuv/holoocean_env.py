"""基于 HoloOcean Torpedo Fossen 动力学的 Gymnasium 训练环境。"""
import contextlib
import io

import numpy as np

from .kinematics import attitude_euler

from .env import DijkstraNavigationEnv
from .holoocean_control import torpedo_fin_commands
from .vehicle_profile import (BOOSTED_PROFILE, DEFAULT_PROFILE,
                              actuator_overrides)


class HoloOceanTorpedoEnv(DijkstraNavigationEnv):
    """直接使用 HoloOcean Torpedo Fossen 模型，避免 Unreal 进程训练开销。

    ``preview.py`` 中的 ``FossenInterface`` 也使用同一个 Torpedo 类，
    因此训练与 HoloOcean 回放共享水动力和执行器模型。
    """

    def __init__(self, max_steps=2_500, obstacle_count=None,
                 goal_guidance=False, capture_assist=False,
                 cruise_assist=True,
                 include_path_observation=False,
                 easy_rehearsal_probability=0.0,
                 dijkstra_resolution=2.0, ticks_per_sec=50,
                 vehicle_profile=DEFAULT_PROFILE,
                 obstacle_count_range=(2, 5)):
        self.vehicle_profile = vehicle_profile
        self.capture_assist = bool(capture_assist)
        self.actuator_overrides = actuator_overrides(vehicle_profile)
        self.ticks_per_sec = int(ticks_per_sec)
        if self.ticks_per_sec <= 0:
            raise ValueError("ticks_per_sec must be positive")
        super().__init__(
            dt=1.0 / self.ticks_per_sec,
            max_steps=max_steps,
            obstacle_count=obstacle_count,
            obstacle_count_range=obstacle_count_range,
            goal_guidance=goal_guidance,
            cruise_assist=cruise_assist,
            include_path_observation=include_path_observation,
            easy_rehearsal_probability=easy_rehearsal_probability,
            dijkstra_resolution=dijkstra_resolution,
            vertical_cost_scale=(3.0 if vehicle_profile == BOOSTED_PROFILE else 1.0),
        )
        self._torpedo = None

    def _reset_torpedo(self):
        from holoocean.fossen_dynamics.torpedo import (
            Torpedo,
            configure_torpedo_from_scenario,
        )

        # Torpedo 构造函数会打印控制模式；训练每回合重建模型时不需要该日志。
        with contextlib.redirect_stdout(io.StringIO()):
            self._torpedo = Torpedo("manualControl")
            # 与 HoloOcean TorpedoAUV 相同的 Fossen Torpedo 模型参数。
            scenario = {"agents": [{"agent_name": "auv0",
                                    "actuator": self.actuator_overrides}]}
            configure_torpedo_from_scenario(scenario, "auv0", self._torpedo)

    def _sample_scene(self):
        super()._sample_scene()
        if self.vehicle_profile == BOOSTED_PROFILE:
            # A faster hull needs a longer braking approach to the goal.
            self.desired_surge = float(self.np_random.uniform(3.0, 4.5))
            self.slowdown_distance = 25.0

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._reset_torpedo()
        return observation, info

    def step(self, action):
        # PPO 输出的是归一化舵角和油门；这里映射到 Torpedo 的物理执行器输入。
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        # 障碍物和目标在回合内静态，复用这张全局代价场；每步从当前位置
        # 重新提取路径前缀，等价于以当前位置作为新的 Dijkstra 起点滚动规划。
        planned_path = self._plan_dijkstra_path(self.state.eta[:3])
        previous_observation = self._observe(planned_path)
        action = self._apply_goal_guidance(action, previous_observation)
        if self.cruise_assist:
            action = self._apply_cruise_floor(action, previous_observation)
        # 新声呐策略默认直接控制三个执行器；仅对照实验才启用旧末端辅助。
        if self.capture_assist:
            action = self._keep_turning_near_goal(action, previous_observation)
        previous_position = self.state.eta[:3].copy()
        actuator = self._torpedo.actuator_parameters
        fin_limit = np.deg2rad(actuator["deltaMax_fin_deg"])
        rpm_limit = float(actuator["nMax"])
        rudder = action[0] * fin_limit
        stern = action[1] * fin_limit
        rpm = (action[2] + 1.0) * 0.5 * rpm_limit
        control = torpedo_fin_commands(rudder, stern, rpm)

        state = self.state
        # Fossen 模型先更新体坐标速度，再用 NED 运动学积分位置和姿态。
        nu_dot, _ = self._torpedo.dynamics(
            state.eta, state.nu, self._torpedo.u_actual, control, self.dt
        )
        state.nu = state.nu + self.dt * nu_dot
        state.eta = attitude_euler(state.eta.copy(), state.nu, self.dt)
        self.steps += 1

        observation = self._observe()
        position = state.eta[:3]
        distance = float(np.linalg.norm(self.goal - position))
        collided = any(np.linalg.norm(position - center) <= radius
                       for center, radius in self.obstacles)
        segment_distance = self._segment_distance(
            self.goal, previous_position, position
        )
        reached = segment_distance <= self.goal_radius

        target_surge = self._target_surge(distance)
        near_goal_ratio = np.clip(
            (self.slowdown_distance - distance) / self.slowdown_distance,
            0.0, 1.0,
        )
        overspeed = max(0.0, state.nu[0] - target_surge)
        arrival_speed_reward = -near_goal_ratio * overspeed ** 2

        # 代价场只提供特权奖励：沿着代价下降方向前进就获得正奖励。
        dijkstra_previous = self._dijkstra_cost(previous_position)
        dijkstra_current = self._dijkstra_cost(position)
        current_outside = self._outside_dijkstra_field(position)
        if np.isfinite(dijkstra_previous) and np.isfinite(dijkstra_current):
            navigation_reward = np.clip(dijkstra_previous - dijkstra_current,
                                        -1.0, 1.0)
        else:
            navigation_reward = -0.1 if current_outside else 0.0
        out_of_bounds = self._outside_planning_bounds(position)
        if out_of_bounds:
            navigation_reward = -1.0

        distance_progress, depth_progress, alignment_progress, speed_shortfall = (
            self._dijkstra_auxiliary_rewards(previous_position, position,
                                             previous_observation, observation,
                                             state.nu[0])
        )
        previous_path_progress, _ = self._path_progress_and_error(
            previous_position, planned_path
        )
        current_path_progress, path_error = self._path_progress_and_error(
            position, planned_path
        )
        path_progress = float(np.clip(
            current_path_progress - previous_path_progress, -0.1, 0.1
        ))
        path_follow_reward = path_progress - 0.02 * float(
            np.clip(path_error, 0.0, 10.0)
        )
        speed_gate = (0.0 if self._is_avoiding(observation) else 1.0) * max(
            0.0, np.cos(float(observation["target"][1]))
        )
        speed_reference = max(target_surge, 0.1)
        cruise_speed_reward = speed_gate * np.clip(
            state.nu[0] / speed_reference, 0.0, 1.0
        )
        (capture_reward, capture_alignment_reward, capture_radial_progress,
         capture_radial_velocity) = self._capture_navigation_rewards(
            previous_position, position, observation, state.nu[0]
        )
        alignment_error = self._navigation_alignment_error(observation)
        navigation_total = (self.dijkstra_reward_scale * navigation_reward
                            + self.dijkstra_distance_progress_weight * distance_progress
                            + self.dijkstra_depth_weight * depth_progress
                            + self.dijkstra_alignment_weight * alignment_progress
                            - self.dijkstra_time_penalty
                            - self.dijkstra_speed_shortfall_weight * speed_shortfall
                            - self.dijkstra_alignment_error_weight * alignment_error)
        reward = (navigation_total
                  + self.dijkstra_cruise_speed_weight * cruise_speed_reward
                  + self.dijkstra_path_follow_weight * path_follow_reward
                  + capture_reward
                  + 0.2 * arrival_speed_reward)
        # 成功、碰撞和越界都是强终止信号，防止失败轨迹继续污染回报。
        if reached:
            reward += self.success_reward
        if collided:
            reward -= self.collision_penalty
        if out_of_bounds:
            reward -= self.out_of_bounds_penalty
        self._previous_action = action.copy()

        info = {
            "distance_to_goal": distance,
            "goal_segment_distance": segment_distance,
            "position": position.copy(),
            "collision": collided,
            "reached": reached,
            "reward_arrival_speed": arrival_speed_reward,
            "target_surge": target_surge,
            "reward_navigation": float(navigation_reward),
            "reward_depth_progress": depth_progress,
            "reward_distance_progress": distance_progress,
            "reward_alignment_progress": alignment_progress,
            "reward_alignment_error": -self.dijkstra_alignment_error_weight * alignment_error,
            "reward_speed_shortfall": -speed_shortfall,
            "reward_cruise_speed": cruise_speed_reward,
            "reward_capture": capture_reward,
            "reward_capture_alignment": capture_alignment_reward,
            "reward_capture_radial_progress": capture_radial_progress,
            "reward_capture_radial_velocity": capture_radial_velocity,
            "reward_path_progress": path_progress,
            "reward_path_follow": path_follow_reward,
            "dijkstra_path_error": path_error,
            "cruise_throttle_floor": self.cruise_min_throttle,
            "reward_time_penalty": -self.dijkstra_time_penalty,
            "reward_terminal_success": self.success_reward if reached else 0.0,
            "reward_terminal_collision": -self.collision_penalty if collided else 0.0,
            "reward_terminal_bounds": -self.out_of_bounds_penalty if out_of_bounds else 0.0,
            "dijkstra_previous": float(dijkstra_previous),
            "dijkstra_current": float(dijkstra_current),
            "dijkstra_outside": bool(current_outside),
            "out_of_bounds": bool(out_of_bounds),
        }
        return observation, float(reward), bool(collided or reached or out_of_bounds), \
            bool(self.steps >= self.max_steps), info
