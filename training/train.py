"""三路声呐 CNN + Dijkstra-v6 PPO 训练入口。"""
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv
from highspeed_uuv.policy import SonarCNNFeaturesExtractor
from highspeed_uuv.vehicle_profile import (
    DEFAULT_PROFILE,
    FIN_MAPPING_VERSION,
    actuator_overrides,
)


def _environment(steps, profile, obstacle_range=(2, 5)):
    return HoloOceanTorpedoEnv(
        max_steps=steps,
        obstacle_count_range=obstacle_range,
        goal_guidance=False,
        capture_assist=False,
        vehicle_profile=profile,
    )


class ObstacleCurriculumCallback(BaseCallback):
    """逐步提高随机障碍物密度，减少策略早期因碰撞/越界而停滞。"""

    def __init__(self, min_obstacles, max_obstacles, schedule_steps, start_step):
        super().__init__()
        self.min_obstacles = int(min_obstacles)
        self.max_obstacles = int(max_obstacles)
        self.schedule_steps = max(1, int(schedule_steps))
        self.start_step = int(start_step)
        self.last_range = None

    def _on_step(self):
        progress = min(1.0, (self.num_timesteps - self.start_step) / self.schedule_steps)
        if progress < 0.35:
            obstacle_range = (0, min(2, self.max_obstacles))
        elif progress < 0.70:
            obstacle_range = (self.min_obstacles, min(5, self.max_obstacles))
        else:
            obstacle_range = (self.min_obstacles, self.max_obstacles)

        if obstacle_range != self.last_range:
            base_env = self.training_env.envs[0].unwrapped
            base_env.obstacle_count_min, base_env.obstacle_count_max = obstacle_range
            self.last_range = obstacle_range
            self.logger.record("curriculum/obstacle_min", obstacle_range[0])
            self.logger.record("curriculum/obstacle_max", obstacle_range[1])
        return True


def main():
    parser = argparse.ArgumentParser(description="Train the three-sonar CNN PPO policy")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--pretrain-timesteps", type=int)
    parser.add_argument("--episode-steps", type=int, default=2_500)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_sonar_dijkstra_v5"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--vehicle-profile", choices=("standard", "boosted"), default=DEFAULT_PROFILE)
    parser.add_argument("--min-obstacles", type=int, default=2)
    parser.add_argument("--max-obstacles", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.episode_steps <= 0 or args.n_steps <= 1:
        parser.error("timesteps, episode-steps, and n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps or args.n_steps % args.batch_size:
        parser.error("batch-size must be a positive divisor of n-steps")
    if args.min_obstacles < 0 or args.max_obstacles < args.min_obstacles:
        parser.error("invalid obstacle range")

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "ppo_sonar.zip"
    if model_path.exists() and not args.overwrite:
        if args.resume is None or args.resume.resolve() != model_path.resolve():
            parser.error(f"model exists: {model_path}; use --overwrite or --resume")
    if args.resume is not None:
        config_path = args.resume.parent / "run.json"
        if not args.resume.is_file() or not config_path.is_file():
            parser.error("resume needs an existing model and its run.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("reward_mode") != "dijkstra"
                or config.get("dijkstra_reward_version") != 11
                or config.get("observation_type") != "three_sonar_maps_v1"):
            parser.error("resume model is not compatible with the three-sonar policy")
        if config.get("vehicle_profile") != args.vehicle_profile:
            parser.error("resume vehicle profile differs; start a new run")
        if config.get("fin_mapping_version") != FIN_MAPPING_VERSION:
            parser.error("resume model uses a different fin mapping")

    torch.set_num_threads(2)
    obstacle_range = (args.min_obstacles, args.max_obstacles)
    initial_range = (0, min(2, args.max_obstacles))
    raw_vec_env = DummyVecEnv([lambda: Monitor(
        _environment(args.episode_steps, args.vehicle_profile, initial_range),
        filename=str(args.output / "monitor.csv"),
    )])
    if args.resume:
        vecnorm_path = args.resume.parent / "vecnormalize.pkl"
        if not vecnorm_path.is_file():
            parser.error("resume requires the matching vecnormalize.pkl")
        env = VecNormalize.load(str(vecnorm_path), raw_vec_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(raw_vec_env, training=True, norm_obs=False,
                           norm_reward=True, gamma=0.999, clip_reward=10.0)
    base_env = env.venv.envs[0].unwrapped
    pretrain_env = None
    try:
        pretrain_steps = args.pretrain_timesteps
        if pretrain_steps is None:
            # 无障碍预训练原本也频繁超时/越界；默认改用同一套渐进课程。
            pretrain_steps = 0
        if pretrain_steps < 0 or (pretrain_steps and args.timesteps - pretrain_steps < 1024):
            parser.error("pretraining must be nonnegative and leave 1024 steps for obstacle training")

        if args.resume:
            model = PPO.load(str(args.resume), device="cpu")
            if model.observation_space != env.observation_space:
                parser.error("resume observation space differs from this environment")
            model.set_env(env)
            steps_before = model.num_timesteps
            pretrain_actual = 0
            remaining = args.timesteps
        else:
            if pretrain_steps:
                pretrain_raw_env = DummyVecEnv([lambda: Monitor(
                    _environment(args.episode_steps, args.vehicle_profile, (0, 0)),
                    filename=str(args.output / "pretrain_monitor.csv"),
                )])
                pretrain_env = VecNormalize(
                    pretrain_raw_env, training=True, norm_obs=False,
                    norm_reward=True, gamma=0.999, clip_reward=10.0,
                )
            model = PPO(
                "MultiInputPolicy", pretrain_env or env,
                seed=args.seed, device="cpu", verbose=1,
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
                gae_lambda=0.995, gamma=0.999, ent_coef=0.01,
                vf_coef=1.0,
                policy_kwargs={
                    "net_arch": [128, 128],
                    "features_extractor_class": SonarCNNFeaturesExtractor,
                    "features_extractor_kwargs": {"features_dim": 256},
                },
            )
            steps_before = 0
            if pretrain_env:
                print(f"无障碍目标导航预训练：{pretrain_steps} 步")
                model.learn(total_timesteps=pretrain_steps)
                env.ret_rms = pretrain_env.ret_rms
                model.set_env(env)
            pretrain_actual = model.num_timesteps
            remaining = args.timesteps - pretrain_actual

        model.set_logger(configure(str(args.output / "logs"), ["stdout", "csv"]))
        curriculum = ObstacleCurriculumCallback(
            args.min_obstacles, args.max_obstacles, remaining, model.num_timesteps
        )
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=args.resume is None and pretrain_actual == 0,
            callback=curriculum,
            log_interval=1,
        )
        model.save(str(model_path.with_suffix("")))
        env.save(str(args.output / "vecnormalize.pkl"))
        # 保存足以复现实验及验证策略的环境参数。
        run = {
            "algorithm": "PPO", "timesteps": args.timesteps,
            "timesteps_before_run": steps_before,
            "timesteps_after_run": model.num_timesteps,
            "pretrain_timesteps": pretrain_actual,
            "ppo_gamma": model.gamma, "ppo_gae_lambda": model.gae_lambda,
            "ppo_n_steps": model.n_steps, "resume_from": str(args.resume) if args.resume else None,
            "ppo_value_coefficient": model.vf_coef,
            "reward_normalization": True,
            "terminal_success_reward": base_env.success_reward,
            "terminal_collision_penalty": base_env.collision_penalty,
            "terminal_out_of_bounds_penalty": base_env.out_of_bounds_penalty,
            "cruise_speed_reward_weight": base_env.dijkstra_cruise_speed_weight,
            "path_follow_reward_weight": base_env.dijkstra_path_follow_weight,
            "capture_reward_distance_m": base_env.capture_reward_distance,
            "capture_alignment_weight": base_env.capture_alignment_weight,
            "capture_radial_progress_weight": base_env.capture_radial_progress_weight,
            "capture_radial_velocity_weight": base_env.capture_radial_velocity_weight,
            "cruise_min_throttle": base_env.cruise_min_throttle,
            "avoidance_min_throttle": base_env.avoidance_min_throttle,
            "cruise_min_distance_m": base_env.cruise_min_distance,
            "cruise_floor_enter_speed_mps": base_env.cruise_floor_enter_speed,
            "cruise_floor_release_speed_mps": base_env.cruise_floor_release_speed,
            "sonar_threat_front_m": 18.0,
            "sonar_threat_side_m": 10.0,
            "obstacle_curriculum": [[0, min(2, args.max_obstacles)],
                                     [args.min_obstacles, min(5, args.max_obstacles)],
                                     list(obstacle_range)],
            "episode_steps": args.episode_steps, "seed": args.seed,
            "batch_size": args.batch_size,
            "reward_mode": "dijkstra", "goal_guidance": False,
            "capture_assist": False,
            "observation_type": "three_sonar_maps_v1",
            "sonar_backend": "geometric ray tracing with HoloOcean sensor geometry; no Unreal runtime",
            "sonar_views": {
                "front": {"type": "ImagingSonar", "azimuth_deg": 120,
                          "elevation_deg": 30},
                "left": {"type": "SidescanSonar", "yaw_deg": -90,
                         "azimuth_deg": 120, "elevation_deg": 5},
                "right": {"type": "SidescanSonar", "yaw_deg": 90,
                          "azimuth_deg": 120, "elevation_deg": 5},
            },
            "sonar_range_m": [base_env.sonar.range_min, base_env.sonar.range_max],
            "sonar_shape": list(base_env.sonar.observation_shape),
            "sonar_noise": False,
            "obstacle_count": base_env.obstacle_count,
            "obstacle_count_range": list(obstacle_range),
            "dynamics_backend": "holocean", "vehicle_profile": args.vehicle_profile,
            "fin_mapping_version": FIN_MAPPING_VERSION,
            "max_propeller_rpm": actuator_overrides(args.vehicle_profile).get("nMax", 1525),
            "dynamics_model": "HoloOcean Fossen Torpedo",
            "dijkstra_resolution_m": base_env.dijkstra_resolution,
            "dijkstra_vertical_cost_scale": base_env.vertical_cost_scale,
            "dijkstra_obstacle_margin_m": 1.5,
            "dijkstra_reward_scale": base_env.dijkstra_reward_scale,
            "dijkstra_depth_weight": base_env.dijkstra_depth_weight,
            "dijkstra_alignment_weight": base_env.dijkstra_alignment_weight,
            "dijkstra_time_penalty": base_env.dijkstra_time_penalty,
            "dijkstra_speed_shortfall_weight": base_env.dijkstra_speed_shortfall_weight,
            "dijkstra_distance_progress_weight": base_env.dijkstra_distance_progress_weight,
            "dijkstra_reward_version": base_env.dijkstra_reward_version,
            "out_of_bounds_penalty": base_env.out_of_bounds_penalty,
            "goal_radius_m": base_env.goal_radius,
            "capture_turn_distance_m": base_env.capture_turn_distance,
            "capture_turn_angle_rad": base_env.capture_turn_angle,
            "capture_min_throttle": base_env.capture_min_throttle,
            "capture_steering_gain": base_env.capture_steering_gain,
            "capture_vertical_gain": base_env.capture_vertical_gain,
            "capture_min_rudder": base_env.capture_min_rudder,
            "planning_bounds_m": [bound.tolist() for bound in base_env.planning_bounds],
            "device": "cpu",
        }
        (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        print(f"模型已保存：{model_path}")
    finally:
        if pretrain_env:
            pretrain_env.close()
        env.close()


if __name__ == "__main__":
    main()
