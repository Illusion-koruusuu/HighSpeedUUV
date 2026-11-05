"""三路声呐 CNN + Dijkstra-v6 PPO 训练入口。

纯 PPO 模式默认关闭运行期目标引导、末端捕获和巡航油门补偿，只保留策略网络
从观测到执行器动作的映射。训练阶段使用由近到远、由无障碍到多障碍的课程，
并定期用固定种子做确定性评估，最终回选成功率最高的 checkpoint。
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv
from highspeed_uuv.policy import BoundedActorCriticPolicy, SonarCNNFeaturesExtractor
from highspeed_uuv.vehicle_profile import (
    DEFAULT_PROFILE,
    FIN_MAPPING_VERSION,
    actuator_overrides,
)


ACTION_DISTRIBUTION = "squashed_diag_gaussian"


def _environment(steps, profile, obstacle_range=(2, 5),
                 goal_distance_range=(50.0, 70.0),
                 goal_yaw_limit=np.pi / 4.0,
                 goal_pitch_limit=np.pi / 12.0,
                 goal_guidance=False, capture_assist=False,
                 cruise_assist=False, include_path_observation=True,
                 easy_rehearsal_probability=0.0):
    env = HoloOceanTorpedoEnv(
        max_steps=steps,
        obstacle_count_range=obstacle_range,
        goal_guidance=goal_guidance,
        capture_assist=capture_assist,
        cruise_assist=cruise_assist,
        include_path_observation=include_path_observation,
        easy_rehearsal_probability=easy_rehearsal_probability,
        vehicle_profile=profile,
    )
    env.goal_distance_range = tuple(float(v) for v in goal_distance_range)
    env.goal_yaw_limit = float(goal_yaw_limit)
    env.goal_pitch_limit = float(goal_pitch_limit)
    return env


def _base_training_envs(training_env):
    """Return unwrapped environment instances behind Monitor/VecNormalize."""
    vec_env = training_env.venv if isinstance(training_env, VecNormalize) else training_env
    return [env.unwrapped for env in vec_env.envs]


def _initialize_action_bias(model, throttle):
    """Bias only the initial policy mean toward forward motion.

    With a squashed Gaussian this is the inverse-tanh of the desired normalized
    throttle. The bias is trainable and does not constrain later actions.
    """
    preactivation = math.atanh(float(np.clip(throttle, -0.95, 0.95)))
    bias = model.policy.action_net.bias
    if bias.shape != (3,):
        raise RuntimeError(f"unexpected policy action bias shape: {bias.shape}")
    with torch.no_grad():
        bias.zero_()
        bias[2] = preactivation


def _teacher_action(observation):
    """Path-following teacher used only for behavior-cloning initialization."""
    path_yaw = float(observation["path"][1])
    path_pitch = float(observation["path"][0])
    surge_error = float(observation["target"][2])
    return np.array([
        np.clip(2.5 * path_yaw, -0.95, 0.95),
        np.clip(-2.0 * path_pitch, -0.95, 0.95),
        np.clip(0.65 + 0.25 * surge_error, -0.3, 0.95),
    ], dtype=np.float32)


def _stack_observations(observations):
    return {
        key: torch.as_tensor(
            np.stack([observation[key] for observation in observations]),
            dtype=torch.float32,
        )
        for key in observations[0]
    }


def evaluate_policy(model, eval_env, n_eval_episodes, seed, timestep=0):
    """Run deterministic evaluation and return navigation metrics."""
    model.policy.set_training_mode(False)
    eval_env.seed(int(seed))
    success = 0
    collisions = 0
    out_of_bounds = 0
    final_distances = []
    steps = []
    for _ in range(n_eval_episodes):
        observation = eval_env.reset()
        last_info = {}
        episode_steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, dones, infos = eval_env.step(action)
            episode_steps += 1
            if bool(dones[0]):
                last_info = infos[0]
                break
        success += int(bool(last_info.get("reached", False)))
        collisions += int(bool(last_info.get("collision", False)))
        out_of_bounds += int(bool(last_info.get("out_of_bounds", False)))
        final_distances.append(float(last_info.get("distance_to_goal", np.nan)))
        steps.append(episode_steps)
    model.policy.set_training_mode(True)
    return {
        "timestep": int(timestep),
        "episodes": n_eval_episodes,
        "success_rate": success / n_eval_episodes,
        "collision_rate": collisions / n_eval_episodes,
        "out_of_bounds_rate": out_of_bounds / n_eval_episodes,
        "mean_final_distance_m": float(np.nanmean(final_distances)),
        "mean_steps": float(np.mean(steps)),
    }


def behavior_clone(model, steps, chunk_size, epochs, learning_rate,
                   episode_steps, profile, goal_range, obstacle_range,
                   yaw_limit, pitch_limit, seed):
    """Warm-start the actor from the path-following teacher.

    The teacher is never called during evaluation or deployment. It only creates
    an initial neural policy that PPO subsequently optimizes.
    """
    env = _environment(
        episode_steps, profile, obstacle_range, goal_range,
        yaw_limit, pitch_limit, goal_guidance=False,
        capture_assist=False, cruise_assist=False,
        include_path_observation=True, easy_rehearsal_probability=0.10,
    )
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    observation, _ = env.reset(seed=seed)
    completed = 0
    last_loss = float("nan")
    try:
        while completed < steps:
            current_chunk = min(chunk_size, steps - completed)
            observations = []
            actions = []
            for _ in range(current_chunk):
                action = _teacher_action(observation)
                observations.append(observation)
                actions.append(action)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    observation, _ = env.reset()
            batch_observations = _stack_observations(observations)
            action_targets = torch.as_tensor(np.stack(actions), dtype=torch.float32)
            action_weights = torch.tensor([2.0, 1.0, 0.5])
            model.policy.set_training_mode(True)
            for _ in range(epochs):
                permutation = torch.randperm(current_chunk)
                shuffled = {
                    key: value[permutation]
                    for key, value in batch_observations.items()
                }
                predicted_mean = model.policy.get_distribution(
                    shuffled
                ).distribution.mean
                predicted_actions = torch.tanh(predicted_mean)
                loss = torch.mean(
                    action_weights
                    * (predicted_actions - action_targets[permutation]) ** 2
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                optimizer.step()
                last_loss = float(loss.detach())
            completed += current_chunk
            print(f"Behavior cloning: {completed}/{steps} steps, "
                  f"action MSE={last_loss:.6f}", flush=True)
    finally:
        env.close()
    return steps, last_loss


class GoalObstacleCurriculumCallback(BaseCallback):
    """Increase goal distance and obstacle density as training progresses."""

    def __init__(self, min_obstacles, max_obstacles, goal_distance_range,
                 goal_yaw_limit, goal_pitch_limit, target_total_steps):
        super().__init__()
        self.min_obstacles = int(min_obstacles)
        self.max_obstacles = int(max_obstacles)
        self.final_goal_distance = tuple(float(v) for v in goal_distance_range)
        self.final_yaw_limit = float(goal_yaw_limit)
        self.final_pitch_limit = float(goal_pitch_limit)
        self.target_total_steps = max(1, int(target_total_steps))
        self.last_config = None

    def _stage(self, progress):
        """Return obstacle range, goal range, yaw limit and pitch limit."""
        if progress < 0.15:
            return (0, 0), (18.0, 30.0), np.deg2rad(30.0), np.deg2rad(10.0), 1
        if progress < 0.35:
            return ((0, min(2, self.max_obstacles)), (25.0, 40.0),
                    np.deg2rad(35.0), np.deg2rad(12.0), 2)
        if progress < 0.60:
            return ((min(self.min_obstacles, 3), min(3, self.max_obstacles)),
                    (35.0, 55.0), np.deg2rad(40.0), np.deg2rad(15.0), 3)
        return ((self.min_obstacles, self.max_obstacles),
                self.final_goal_distance, self.final_yaw_limit,
                self.final_pitch_limit, 4)

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.target_total_steps)
        obstacle_range, goal_range, yaw_limit, pitch_limit, stage = self._stage(progress)
        config = (obstacle_range, goal_range, yaw_limit, pitch_limit, stage)
        if config == self.last_config:
            return True

        for env in _base_training_envs(self.training_env):
            env.obstacle_count_min, env.obstacle_count_max = obstacle_range
            env.goal_distance_range = goal_range
            env.goal_yaw_limit = yaw_limit
            env.goal_pitch_limit = pitch_limit
        self.last_config = config
        self.logger.record("curriculum/stage", stage)
        self.logger.record("curriculum/obstacle_min", obstacle_range[0])
        self.logger.record("curriculum/obstacle_max", obstacle_range[1])
        self.logger.record("curriculum/goal_min_m", goal_range[0])
        self.logger.record("curriculum/goal_max_m", goal_range[1])
        return True


class BestSuccessEvalCallback(BaseCallback):
    """Evaluate deterministic success and preserve the best pure-PPO model."""

    def __init__(self, eval_env, eval_freq, n_eval_episodes,
                 best_model_path, best_info_path, seed=0):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.best_model_path = best_model_path
        self.best_info_path = best_info_path
        self.seed = int(seed)
        self.last_eval_step = 0
        self.best_key = None
        self.best_info = None
        if best_info_path.is_file():
            try:
                previous = json.loads(best_info_path.read_text(encoding="utf-8"))
                self.best_info = previous
                self.best_key = (
                    float(previous["success_rate"]),
                    -float(previous["mean_final_distance_m"]),
                    -float(previous["collision_rate"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.best_key = None
                self.best_info = None

    def _evaluate(self):
        return evaluate_policy(
            self.model, self.eval_env, self.n_eval_episodes, self.seed,
            timestep=self.num_timesteps,
        )

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        result = self._evaluate()
        self.logger.record("eval/success_rate", result["success_rate"])
        self.logger.record("eval/collision_rate", result["collision_rate"])
        self.logger.record("eval/out_of_bounds_rate", result["out_of_bounds_rate"])
        self.logger.record("eval/mean_final_distance_m",
                           result["mean_final_distance_m"])
        key = (result["success_rate"], -result["mean_final_distance_m"],
               -result["collision_rate"])
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.best_info = result
            self.model.save(str(self.best_model_path.with_suffix("")))
            self.best_info_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(f"New best pure-PPO checkpoint at {self.num_timesteps} steps: "
                  f"success={result['success_rate']:.0%}, "
                  f"mean_final_distance={result['mean_final_distance_m']:.2f} m")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Train a pure three-sonar CNN PPO policy with curriculum"
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="新训练总步数；续训时表示本次额外增加的步数")
    parser.add_argument("--episode-steps", type=int, default=2_500)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--log-std-init", type=float, default=-1.0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.995)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--initial-throttle", type=float, default=0.6,
                        help="仅初始化策略动作均值，后续完全可学习")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--bc-steps", type=int, default=5_000,
                        help="教师行为克隆预热步数；0 表示完全从零 PPO 训练")
    parser.add_argument("--bc-chunk-size", type=int, default=1_000)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-finetune-learning-rate", type=float, default=1e-4,
                        help="BC 初始化后 PPO 微调学习率")
    parser.add_argument("--path-observation", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="把局部 Dijkstra 路径方向加入策略观测")
    parser.add_argument("--rehearsal-probability", type=float, default=0.25,
                        help="训练时随机抽取短距离无障碍复习回合的概率")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/ppo_sonar_pure"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--vehicle-profile", choices=("standard", "boosted"),
                        default=DEFAULT_PROFILE)
    parser.add_argument("--min-obstacles", type=int, default=2)
    parser.add_argument("--max-obstacles", type=int, default=5)
    parser.add_argument("--goal-distance-min", type=float, default=50.0)
    parser.add_argument("--goal-distance-max", type=float, default=70.0)
    parser.add_argument("--goal-yaw-limit-deg", type=float, default=45.0)
    parser.add_argument("--goal-pitch-limit-deg", type=float, default=15.0)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="BC 预热后默认直接在全难度分布上做 PPO 微调")
    parser.add_argument("--goal-guidance", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="纯 PPO 默认关闭；仅用于对照实验")
    parser.add_argument("--capture-assist", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="纯 PPO 默认关闭；仅用于对照实验")
    parser.add_argument("--cruise-assist", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="纯 PPO 默认关闭；仅用于对照实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.timesteps <= 0 or args.episode_steps <= 0 or args.n_steps <= 1:
        parser.error("timesteps, episode-steps, and n-steps must be positive")
    if (args.batch_size <= 0 or args.batch_size > args.n_steps
            or args.n_steps % args.batch_size):
        parser.error("batch-size must be a positive divisor of n-steps")
    if args.min_obstacles < 0 or args.max_obstacles < args.min_obstacles:
        parser.error("invalid obstacle range")
    if not 0.0 < args.goal_distance_min <= args.goal_distance_max:
        parser.error("invalid final goal distance range")
    if args.eval_freq <= 0 or args.eval_episodes <= 0:
        parser.error("eval-freq and eval-episodes must be positive")
    if (args.bc_steps < 0 or args.bc_chunk_size <= 0 or args.bc_epochs <= 0
            or args.bc_learning_rate <= 0 or args.bc_finetune_learning_rate <= 0):
        parser.error("BC parameters must be nonnegative/positive as appropriate")
    if not -0.95 <= args.initial_throttle <= 0.95:
        parser.error("initial-throttle must be in [-0.95, 0.95]")
    if not 0.0 <= args.rehearsal_probability <= 1.0:
        parser.error("rehearsal-probability must be in [0, 1]")

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "ppo_sonar.zip"
    final_model_path = args.output / "ppo_sonar_final.zip"
    best_model_path = args.output / "best_model.zip"
    best_info_path = args.output / "best_eval.json"
    if model_path.exists() and not args.overwrite:
        same_model = (args.resume is not None
                      and args.resume.resolve() == model_path.resolve())
        if not same_model:
            parser.error(f"model exists: {model_path}; use --overwrite or --resume")

    resume_file = None
    if args.resume is not None:
        resume_file = args.resume if args.resume.suffix == ".zip" else args.resume.with_suffix(".zip")
        config_path = resume_file.parent / "run.json"
        if not resume_file.is_file() or not config_path.is_file():
            parser.error("resume needs an existing model and its run.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("reward_mode") != "dijkstra"
                or config.get("dijkstra_reward_version") != 12
                or config.get("observation_type") not in (
                    "three_sonar_maps_v1", "three_sonar_path_v2")):
            parser.error("resume model is incompatible with this environment")
        if config.get("vehicle_profile") != args.vehicle_profile:
            parser.error("resume vehicle profile differs; start a new run")
        if config.get("fin_mapping_version") != FIN_MAPPING_VERSION:
            parser.error("resume model uses a different fin mapping")
        if config.get("policy_action_distribution") != ACTION_DISTRIBUTION:
            parser.error("resume model predates bounded action training; start a new run")

    if resume_file is None:
        # A fresh run must not reuse a best checkpoint left by an older run.
        for stale_path in (model_path, best_model_path, best_info_path, final_model_path,
                           args.output / "vecnormalize_final.pkl",
                           args.output / "vecnormalize.pkl"):
            if stale_path.exists():
                stale_path.unlink()

    torch.set_num_threads(2)
    logger = configure(str(args.output / "logs"), ["stdout", "csv"])
    final_obstacle_range = (args.min_obstacles, args.max_obstacles)
    final_goal_range = (args.goal_distance_min, args.goal_distance_max)
    final_yaw_limit = np.deg2rad(args.goal_yaw_limit_deg)
    final_pitch_limit = np.deg2rad(args.goal_pitch_limit_deg)
    initial_goal_range = ((18.0, 30.0) if args.curriculum else final_goal_range)
    initial_obstacle_range = ((0, 0) if args.curriculum else final_obstacle_range)

    raw_env = DummyVecEnv([lambda: Monitor(
        _environment(
            args.episode_steps, args.vehicle_profile, initial_obstacle_range,
            initial_goal_range,
            np.deg2rad(30.0) if args.curriculum else final_yaw_limit,
            np.deg2rad(10.0) if args.curriculum else final_pitch_limit,
            args.goal_guidance, args.capture_assist, args.cruise_assist,
            args.path_observation, args.rehearsal_probability,
        ),
        filename=str(args.output / "monitor.csv"),
    )])
    if resume_file is not None:
        vecnorm_path = resume_file.parent / "vecnormalize.pkl"
        if not vecnorm_path.is_file():
            parser.error("resume requires the matching vecnormalize.pkl")
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(raw_env, training=True, norm_obs=False,
                           norm_reward=True, gamma=args.gamma, clip_reward=10.0)

    eval_raw_env = DummyVecEnv([lambda: _environment(
        args.episode_steps, args.vehicle_profile, final_obstacle_range,
        final_goal_range, final_yaw_limit, final_pitch_limit,
        args.goal_guidance, args.capture_assist, args.cruise_assist,
        args.path_observation, 0.0,
    )])
    base_env = env.venv.envs[0].unwrapped
    try:
        if resume_file is not None:
            model = PPO.load(str(resume_file), device="cpu")
            if model.observation_space != env.observation_space:
                parser.error("resume observation space differs from this environment")
            model.set_env(env)
            steps_before = model.num_timesteps
            target_total_steps = steps_before + args.timesteps
            reset_num_timesteps = False
        else:
            model = PPO(
                BoundedActorCriticPolicy,
                env,
                seed=args.seed,
                device="cpu",
                verbose=1,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gae_lambda=args.gae_lambda,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                vf_coef=1.0,
                max_grad_norm=0.5,
                target_kl=args.target_kl,
                policy_kwargs={
                    "net_arch": [256, 256],
                    "features_extractor_class": SonarCNNFeaturesExtractor,
                    "features_extractor_kwargs": {"features_dim": 256},
                    "log_std_init": args.log_std_init,
                },
            )
            _initialize_action_bias(model, args.initial_throttle)
            steps_before = 0
            target_total_steps = args.timesteps
            reset_num_timesteps = True

        bc_steps_actual = 0
        bc_final_loss = None
        bc_eval = None
        if resume_file is None and args.bc_steps > 0:
            bc_steps_actual, bc_final_loss = behavior_clone(
                model, args.bc_steps, args.bc_chunk_size, args.bc_epochs,
                args.bc_learning_rate, args.episode_steps, args.vehicle_profile,
                final_goal_range, final_obstacle_range, final_yaw_limit,
                final_pitch_limit, args.seed,
            )
            model.save(str((args.output / "bc_model").with_suffix("")))
            bc_eval = evaluate_policy(
                model, eval_raw_env, args.eval_episodes, args.seed, timestep=0
            )
            print(f"BC policy evaluation: success={bc_eval['success_rate']:.0%}, "
                  f"mean_final_distance={bc_eval['mean_final_distance_m']:.2f} m")
            if bc_eval["success_rate"] > 0.0:
                model.save(str(best_model_path.with_suffix("")))
                best_info_path.write_text(
                    json.dumps(bc_eval, indent=2) + "\n", encoding="utf-8"
                )
            model.learning_rate = args.bc_finetune_learning_rate
            for group in model.policy.optimizer.param_groups:
                group["lr"] = args.bc_finetune_learning_rate

        callbacks = []
        if args.curriculum:
            callbacks.append(GoalObstacleCurriculumCallback(
                args.min_obstacles, args.max_obstacles, final_goal_range,
                final_yaw_limit, final_pitch_limit, target_total_steps,
            ))
        callbacks.append(BestSuccessEvalCallback(
            eval_raw_env, args.eval_freq, args.eval_episodes,
            best_model_path, best_info_path, seed=args.seed,
        ))
        model.set_logger(logger)
        model.learn(
            total_timesteps=target_total_steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )

        model.save(str(final_model_path.with_suffix("")))
        env.save(str(args.output / "vecnormalize_final.pkl"))
        if best_model_path.is_file():
            shutil.copyfile(best_model_path, model_path)
            selected_checkpoint = "best_model.zip"
        else:
            shutil.copyfile(final_model_path, model_path)
            selected_checkpoint = "ppo_sonar_final.zip"
        env.save(str(args.output / "vecnormalize.pkl"))

        best_info = None
        if best_info_path.is_file():
            best_info = json.loads(best_info_path.read_text(encoding="utf-8"))
        run = {
            "algorithm": "PPO",
            "policy_mode": ("bc_initialized_pure_ppo" if bc_steps_actual
                            else "pure_ppo"),
            "policy_action_distribution": ACTION_DISTRIBUTION,
            "timesteps": args.timesteps,
            "timesteps_before_run": steps_before,
            "timesteps_after_run": model.num_timesteps,
            "selected_checkpoint": selected_checkpoint,
            "best_eval": best_info,
            "bc_steps": bc_steps_actual,
            "bc_final_action_mse": bc_final_loss,
            "bc_eval": bc_eval,
            "bc_teacher": "path-yaw/pitch proportional controller",
            "resume_from": str(resume_file) if resume_file else None,
            "episode_steps": args.episode_steps,
            "seed": args.seed,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "bc_finetune_learning_rate": (
                args.bc_finetune_learning_rate if bc_steps_actual else None
            ),
            "ppo_gamma": args.gamma,
            "ppo_gae_lambda": args.gae_lambda,
            "ppo_ent_coef": args.ent_coef,
            "ppo_log_std_init": args.log_std_init if resume_file is None else None,
            "ppo_target_kl": args.target_kl,
            "initial_throttle_action": args.initial_throttle if resume_file is None else None,
            "reward_normalization": True,
            "reward_mode": "dijkstra",
            "dijkstra_reward_version": base_env.dijkstra_reward_version,
            "observation_type": ("three_sonar_path_v2" if args.path_observation
                                 else "three_sonar_maps_v1"),
            "include_path_observation": args.path_observation,
            "path_observation": ("local Dijkstra path pitch/yaw, lookahead distance, "
                                 "remaining cost"),
            "easy_rehearsal_probability": args.rehearsal_probability,
            "policy_net_arch": [256, 256],
            "goal_guidance": args.goal_guidance,
            "capture_assist": args.capture_assist,
            "cruise_assist": args.cruise_assist,
            "curriculum": args.curriculum,
            "curriculum_stages": {
                "0-15%": {"obstacles": [0, 0], "goal_distance_m": [18, 30]},
                "15-35%": {"obstacles": [0, min(2, args.max_obstacles)],
                           "goal_distance_m": [25, 40]},
                "35-60%": {"obstacles": [min(args.min_obstacles, 3),
                                          min(3, args.max_obstacles)],
                           "goal_distance_m": [35, 55]},
                "60-100%": {"obstacles": list(final_obstacle_range),
                            "goal_distance_m": list(final_goal_range)},
            } if args.curriculum else None,
            "obstacle_count_range": list(final_obstacle_range),
            "goal_distance_range_m": list(final_goal_range),
            "goal_yaw_limit_deg": args.goal_yaw_limit_deg,
            "goal_pitch_limit_deg": args.goal_pitch_limit_deg,
            "dynamics_backend": "holocean",
            "vehicle_profile": args.vehicle_profile,
            "fin_mapping_version": FIN_MAPPING_VERSION,
            "max_propeller_rpm": actuator_overrides(args.vehicle_profile).get("nMax", 1525),
            "dynamics_model": "HoloOcean Fossen Torpedo",
            "dijkstra_resolution_m": base_env.dijkstra_resolution,
            "dijkstra_vertical_cost_scale": base_env.vertical_cost_scale,
            "dijkstra_obstacle_margin_m": 1.5,
            "dijkstra_reward_scale": base_env.dijkstra_reward_scale,
            "dijkstra_depth_weight": base_env.dijkstra_depth_weight,
            "dijkstra_alignment_weight": base_env.dijkstra_alignment_weight,
            "dijkstra_alignment_error_weight": base_env.dijkstra_alignment_error_weight,
            "dijkstra_time_penalty": base_env.dijkstra_time_penalty,
            "dijkstra_speed_shortfall_weight": base_env.dijkstra_speed_shortfall_weight,
            "dijkstra_distance_progress_weight": base_env.dijkstra_distance_progress_weight,
            "out_of_bounds_penalty": base_env.out_of_bounds_penalty,
            "goal_radius_m": base_env.goal_radius,
            "planning_bounds_m": [bound.tolist() for bound in base_env.planning_bounds],
            "device": "cpu",
        }
        (args.output / "run.json").write_text(
            json.dumps(run, indent=2) + "\n", encoding="utf-8"
        )
        print(f"训练完成，主模型已保存为 {model_path} ({selected_checkpoint})")
    finally:
        eval_raw_env.close()
        env.close()


if __name__ == "__main__":
    main()
