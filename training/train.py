"""HoloOcean Torpedo 模型上的 Dijkstra-v5 PPO 训练入口。"""
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv
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
        vehicle_profile=profile,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the HoloOcean Dijkstra-v5 policy")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--pretrain-timesteps", type=int)
    parser.add_argument("--episode-steps", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_dijkstra_v6"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--vehicle-profile", choices=("standard", "boosted"), default=DEFAULT_PROFILE)
    parser.add_argument("--min-obstacles", type=int, default=2)
    parser.add_argument("--max-obstacles", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.episode_steps <= 0:
        parser.error("timesteps and episode-steps must be positive")
    if args.min_obstacles < 0 or args.max_obstacles < args.min_obstacles:
        parser.error("invalid obstacle range")

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "ppo_dijkstra.zip"
    if model_path.exists() and not args.overwrite:
        if args.resume is None or args.resume.resolve() != model_path.resolve():
            parser.error(f"model exists: {model_path}; use --overwrite or --resume")
    if args.resume is not None:
        config_path = args.resume.parent / "run.json"
        if not args.resume.is_file() or not config_path.is_file():
            parser.error("resume needs an existing model and its run.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("reward_mode") != "dijkstra" or config.get("dijkstra_reward_version") != 5:
            parser.error("resume model is not compatible with Dijkstra reward v5")
        if config.get("vehicle_profile") != args.vehicle_profile:
            parser.error("resume vehicle profile differs; start a new run")
        if config.get("fin_mapping_version") != FIN_MAPPING_VERSION:
            parser.error("resume model uses a different fin mapping")

    torch.set_num_threads(2)
    obstacle_range = (args.min_obstacles, args.max_obstacles)
    base_env = _environment(args.episode_steps, args.vehicle_profile, obstacle_range)
    env = Monitor(base_env, filename=str(args.output / "monitor.csv"))
    pretrain_env = None
    try:
        pretrain_steps = args.pretrain_timesteps
        if pretrain_steps is None:
            pretrain_steps = args.timesteps // 5 if args.timesteps >= 2048 else 0
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
                pretrain_env = Monitor(
                    _environment(args.episode_steps, args.vehicle_profile, (0, 0)),
                    filename=str(args.output / "pretrain_monitor.csv"),
                )
            model = PPO(
                "MultiInputPolicy", pretrain_env or env,
                seed=args.seed, device="cpu", verbose=1,
                n_steps=1024, batch_size=256, n_epochs=10,
                gae_lambda=0.98, gamma=0.999, ent_coef=0.01,
                policy_kwargs={"net_arch": [128, 128]},
            )
            steps_before = 0
            if pretrain_env:
                print(f"无障碍目标导航预训练：{pretrain_steps} 步")
                model.learn(total_timesteps=pretrain_steps)
                model.set_env(env)
            pretrain_actual = model.num_timesteps
            remaining = args.timesteps - pretrain_actual

        model.learn(total_timesteps=remaining, reset_num_timesteps=args.resume is None and pretrain_actual == 0)
        model.save(str(model_path.with_suffix("")))
        # 保存足以复现实验及验证策略的环境参数。
        run = {
            "algorithm": "PPO", "timesteps": args.timesteps,
            "timesteps_before_run": steps_before,
            "timesteps_after_run": model.num_timesteps,
            "pretrain_timesteps": pretrain_actual,
            "ppo_gamma": model.gamma, "ppo_gae_lambda": model.gae_lambda,
            "ppo_n_steps": model.n_steps, "resume_from": str(args.resume) if args.resume else None,
            "episode_steps": args.episode_steps, "seed": args.seed,
            "reward_mode": "dijkstra", "goal_guidance": False,
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
