"""Replay a saved policy and plot its path against the training obstacles."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
from stable_baselines3 import PPO

from highspeed_uuv.dijkstra import build_cost_field
from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv
from highspeed_uuv.policy import SonarCNNFeaturesExtractor  # 确保可反序列化 PPO CNN
from highspeed_uuv.vehicle_profile import (DEFAULT_PROFILE, FIN_MAPPING_VERSION,
                                           STANDARD_PROFILE)


ASSIST_ATTRIBUTES = {
    'capture_turn_distance_m': 'capture_turn_distance',
    'capture_turn_angle_rad': 'capture_turn_angle',
    'capture_min_throttle': 'capture_min_throttle',
    'capture_steering_gain': 'capture_steering_gain',
    'capture_vertical_gain': 'capture_vertical_gain',
    'capture_min_rudder': 'capture_min_rudder',
}


def make_env(episode_steps, goal_guidance, vehicle_profile=DEFAULT_PROFILE,
             obstacle_count_range=(2, 5), capture_assist=False,
             cruise_assist=True,
             include_path_observation=False,
             assist_overrides=None):
    kwargs = dict(
        max_steps=episode_steps,
        obstacle_count_range=obstacle_count_range,
        goal_guidance=goal_guidance,
        capture_assist=capture_assist,
        cruise_assist=cruise_assist,
        include_path_observation=include_path_observation,
    )
    kwargs['vehicle_profile'] = vehicle_profile
    env = HoloOceanTorpedoEnv(**kwargs)
    for key, value in (assist_overrides or {}).items():
        if key in ('goal_guidance', 'capture_assist'):
            continue
        attribute = ASSIST_ATTRIBUTES.get(key)
        if attribute is None:
            raise ValueError(f'unknown evaluation assist parameter: {key}')
        setattr(env, attribute, value)
    return env


def load_run_config(model_path):
    """读取训练配置，使评估环境与模型训练环境保持一致。"""
    path = model_path.parent / 'run.json'
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def build_reference_path(env, resolution=2.0, max_points=2000):
    """构建 Dijkstra 代价场，并沿代价下降方向提取参考路径。"""
    if env._cost_field is not None:
        costs = env._cost_field
        origin = env._cost_origin
        resolution = env._cost_resolution
        occupancy = env._cost_occupancy
    else:
        costs, origin, resolution, occupancy = build_cost_field(
            env.goal,
            env.obstacles,
            start=env.state.eta[:3],
            resolution=resolution,
            obstacle_margin=1.5,
            goal_radius=env.goal_radius,
            planning_bounds=env.planning_bounds,
            vertical_cost_scale=env.vertical_cost_scale,
        )
    shape = np.asarray(costs.shape, dtype=int)
    current = tuple(np.rint(
        (env.state.eta[:3] - origin) / resolution
    ).astype(int).clip(0, shape - 1))
    path = [origin + np.asarray(current) * resolution]
    offsets = [
        (dx, dy, dz) for dx in (-1, 0, 1)
        for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    for _ in range(max_points):
        if not np.isfinite(costs[current]) or costs[current] <= 1e-5:
            break
        candidates = []
        for dx, dy, dz in offsets:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if any(index < 0 or index >= shape[axis]
                   for axis, index in enumerate(neighbor)):
                continue
            if occupancy[neighbor] or not np.isfinite(costs[neighbor]):
                continue
            candidates.append((float(costs[neighbor]), neighbor))
        if not candidates:
            break
        next_cost, next_index = min(candidates, key=lambda item: item[0])
        if next_cost >= float(costs[current]) - 1e-6:
            break
        current = next_index
        path.append(origin + np.asarray(current) * resolution)
    return np.asarray(path), origin, origin + (shape - 1) * resolution


def draw_grid_bounds_3d(ax, lower, upper):
    """用半透明线框显示 Dijkstra 三维栅格规划范围。"""
    corners = np.array([
        [x, y, z] for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ])
    edges = ((0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
             (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7))
    for start, end in edges:
        ax.plot(*zip(corners[start], corners[end]), color='purple',
                alpha=0.28, linewidth=1.0, linestyle='--')


def draw_grid_bounds_2d(ax, lower, upper):
    """显示规划栅格在水平面的半透明投影。"""
    ax.fill(
        [lower[0], upper[0], upper[0], lower[0]],
        [lower[1], lower[1], upper[1], upper[1]],
        color='mediumpurple', alpha=0.08, label='Dijkstra grid area',
    )
    ax.plot(
        [lower[0], upper[0], upper[0], lower[0], lower[0]],
        [lower[1], lower[1], upper[1], upper[1], lower[1]],
        color='purple', alpha=0.35, linestyle='--', linewidth=1.0,
    )


def draw_sphere(ax, center, radius):
    """绘制真球坐标；显示比例由 configure_3d_axes 统一控制。"""
    azimuth = np.linspace(0.0, 2.0 * np.pi, 24)
    elevation = np.linspace(0.0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(azimuth), np.sin(elevation))
    y = center[1] + radius * np.outer(np.sin(azimuth), np.sin(elevation))
    z = center[2] + radius * np.outer(np.ones_like(azimuth), np.cos(elevation))
    ax.plot_wireframe(x, y, z, color='orange', alpha=0.5, linewidth=0.6)


def configure_3d_axes(ax, path, goal, obstacles, reference_path=None,
                       grid_bounds=None):
    """让三个坐标轴使用同一显示尺度，避免球体被拉扁。"""
    points = [path, np.asarray([goal])]
    if reference_path is not None:
        points.append(reference_path)
    if grid_bounds is not None:
        points.extend(grid_bounds)
    for center, radius in obstacles:
        points.append(np.asarray([center - radius, center + radius]))
    bounds = np.vstack(points)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    center = (lower + upper) / 2.0
    half_extent = max((upper - lower).max() / 2.0, 1.0) * 1.08
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
    ax.set_zlim(center[2] + half_extent, center[2] - half_extent)
    ax.set_box_aspect((1, 1, 1))


def draw_evaluation(fig, ax3d, ax2d, path, env, total_reward, distance, steps,
                    reference_path=None, grid_bounds=None):
    """刷新实时评估窗口，并保持三维球体的等比例显示。"""
    ax3d.clear()
    ax2d.clear()
    ax3d.plot(path[:, 0], path[:, 1], path[:, 2], color='navy', lw=2,
              label='PPO trajectory')
    if reference_path is not None:
        ax3d.plot(reference_path[:, 0], reference_path[:, 1], reference_path[:, 2],
                  color='limegreen', lw=1.8, linestyle='--',
                  label='Dijkstra reference')
    ax3d.scatter(*path[0], color='green', s=50, label='start')
    ax3d.scatter(*env.goal, color='red', marker='*', s=180, label='goal')
    for center, radius in env.obstacles:
        draw_sphere(ax3d, center, radius)
    if grid_bounds is not None:
        draw_grid_bounds_3d(ax3d, *grid_bounds)
    configure_3d_axes(ax3d, path, env.goal, env.obstacles,
                       reference_path, grid_bounds)
    ax3d.set_xlabel('North x (m)')
    ax3d.set_ylabel('East y (m)')
    ax3d.set_zlabel('Depth z (m, positive down)')
    ax3d.view_init(elev=24, azim=-62)
    ax3d.set_title(f'3-D trajectory | step={steps}')
    ax3d.legend(loc='upper left')

    ax2d.plot(path[:, 0], path[:, 1], color='navy', lw=2,
              label='PPO trajectory')
    if reference_path is not None:
        ax2d.plot(reference_path[:, 0], reference_path[:, 1],
                  color='limegreen', lw=1.8, linestyle='--',
                  label='Dijkstra reference')
    ax2d.scatter(path[0, 0], path[0, 1], color='green', label='start')
    ax2d.scatter(env.goal[0], env.goal[1], color='red', marker='*', s=160,
                 label='goal')
    for center, radius in env.obstacles:
        ax2d.add_patch(Circle(center[:2], radius, alpha=0.35, color='orange'))
    if grid_bounds is not None:
        draw_grid_bounds_2d(ax2d, *grid_bounds)
    ax2d.set_aspect('equal', adjustable='datalim')
    ax2d.set_xlabel('North x (m)')
    ax2d.set_ylabel('East y (m)')
    ax2d.set_title(f'Top view | reward={total_reward:.2f}; distance={distance:.1f} m')
    ax2d.grid(True)
    ax2d.legend(loc='best')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def run_episode(env, model, seed, live=False, render_every=5, figure=None):
    """在一个固定随机种子场景中评估策略，并返回完整统计信息。"""
    observation, _ = env.reset(seed=seed)
    path = [env.state.eta[:3].copy()]
    reference_path, grid_lower, grid_upper = build_reference_path(env)
    grid_bounds = (grid_lower, grid_upper)
    total_reward = 0.0
    min_distance = float(np.linalg.norm(env.goal - path[-1]))
    speed_at_min_distance = float(env.state.nu[0])
    surge_sum = 0.0
    throttle_sum = 0.0
    full_throttle_steps = 0
    if live:
        fig, ax3d, ax2d = figure
        draw_evaluation(fig, ax3d, ax2d, np.asarray(path), env, 0.0,
                        float(np.linalg.norm(env.goal - path[-1])), 0,
                        reference_path, grid_bounds)
        plt.pause(0.001)

    for step in range(1, env.max_steps + 1):
        action, _ = model.predict(observation, deterministic=True)
        throttle = float(np.clip((action[2] + 1.0) * 0.5, 0.0, 1.0))
        throttle_sum += throttle
        full_throttle_steps += throttle >= 0.99
        observation, reward, terminated, truncated, info = env.step(action)
        surge_sum += float(env.state.nu[0])
        total_reward += reward
        path.append(env.state.eta[:3].copy())
        if info['distance_to_goal'] < min_distance:
            min_distance = float(info['distance_to_goal'])
            speed_at_min_distance = float(env.state.nu[0])
        if live and step % render_every == 0:
            fig, ax3d, ax2d = figure
            draw_evaluation(fig, ax3d, ax2d, np.asarray(path), env,
                            total_reward, info['distance_to_goal'], step,
                            reference_path, grid_bounds)
            plt.pause(0.001)
        if terminated or truncated:
            break

    path = np.asarray(path)
    path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    final_error = env.goal - path[-1]
    result = {
        'seed': seed,
        'success': int(info['reached']),
        'collision': int(info['collision']),
        'out_of_bounds': int(info.get('out_of_bounds', False)),
        'timeout': int(truncated and not terminated),
        'reward': float(total_reward),
        'steps': step,
        'final_distance_m': float(info['distance_to_goal']),
        'final_horizontal_error_m': float(np.linalg.norm(final_error[:2])),
        'final_depth_error_m': float(abs(final_error[2])),
        'min_distance_m': min_distance,
        'speed_at_min_distance_mps': speed_at_min_distance,
        'path_length_m': path_length,
        'mean_surge_mps': surge_sum / step,
        'mean_ground_speed_mps': path_length / (step * env.dt),
        'mean_throttle_fraction': throttle_sum / step,
        'full_throttle_fraction': full_throttle_steps / step,
        'desired_surge_mps': float(env.desired_surge),
        'obstacle_count': len(env.obstacles),
    }
    return result, path, info, reference_path, grid_bounds


def write_metrics(path, results):
    """保存每回合记录和最终汇总，便于后续在表格软件中统计。"""
    success_rate = float(np.mean([row['success'] for row in results]))
    collision_rate = float(np.mean([row['collision'] for row in results]))
    out_of_bounds_rate = float(np.mean([row['out_of_bounds'] for row in results]))
    timeout_rate = float(np.mean([row['timeout'] for row in results]))
    means = {
        key: float(np.mean([row[key] for row in results]))
        for key in ('reward', 'steps', 'final_distance_m',
                    'final_horizontal_error_m', 'final_depth_error_m',
                    'min_distance_m',
                    'speed_at_min_distance_mps', 'path_length_m',
                    'mean_surge_mps', 'mean_ground_speed_mps',
                    'mean_throttle_fraction', 'full_throttle_fraction')
    }
    fields = [
        'record_type', 'episode', 'seed', 'success', 'collision', 'out_of_bounds',
        'timeout',
        'reward', 'steps', 'final_distance_m', 'final_horizontal_error_m',
        'final_depth_error_m', 'min_distance_m',
        'speed_at_min_distance_mps', 'path_length_m', 'mean_surge_mps',
        'mean_ground_speed_mps', 'mean_throttle_fraction',
        'full_throttle_fraction',
        'desired_surge_mps', 'obstacle_count', 'success_rate',
        'collision_rate', 'out_of_bounds_rate', 'timeout_rate',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode, row in enumerate(results, start=1):
            writer.writerow({'record_type': 'episode', 'episode': episode, **row})
        writer.writerow({
            'record_type': 'summary',
            'episode': len(results),
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'out_of_bounds_rate': out_of_bounds_rate,
            'timeout_rate': timeout_rate,
            **means,
        })
    return success_rate, collision_rate, out_of_bounds_rate, timeout_rate, means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path, help='Saved PPO .zip file')
    parser.add_argument('--output', type=Path, default=Path('runs/ppo_test'),
                        help='输出目录；保存 evaluation.png 和 evaluation_metrics.csv')
    parser.add_argument('--episode-steps', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--obstacle-count', type=int,
                        help='兼容旧配置：固定静态障碍物数量')
    parser.add_argument('--min-obstacles', type=int, default=None)
    parser.add_argument('--max-obstacles', type=int, default=None)
    parser.add_argument('--goal-guidance', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='覆盖 run.json 中的目标引导开关')
    parser.add_argument('--capture-assist', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='覆盖 run.json 中的末端捕获辅助开关')
    parser.add_argument('--cruise-assist', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='覆盖 run.json 中的低速推进下限辅助开关')
    parser.add_argument('--episodes', type=int, default=1,
                        help='测试回合数；依次使用 seed 到 seed + episodes - 1')
    parser.add_argument('--csv', type=Path,
                        help='覆盖默认 CSV 路径（默认输出目录/evaluation_metrics.csv）')
    parser.add_argument('--live', action='store_true',
                        help='评估过程中实时更新三维轨迹和俯视图')
    parser.add_argument('--render-every', type=int, default=50,
                        help='实时可视化每隔多少个环境步刷新一次')
    parser.add_argument('--show', action='store_true',
                        help='在保存图片后显示评估地图窗口')
    args = parser.parse_args()
    if args.episode_steps <= 0 or args.episodes <= 0 or args.render_every <= 0:
        parser.error('episode-steps, episodes, and render-every must be positive')

    config = load_run_config(args.model)
    observation_type = config.get('observation_type')
    if observation_type not in ('three_sonar_maps_v1', 'three_sonar_path_v2'):
        parser.error('model run.json is not a supported three-sonar CNN run')
    include_path_observation = config.get(
        'include_path_observation', observation_type == 'three_sonar_path_v2'
    )
    vehicle_profile = config.get('vehicle_profile') or STANDARD_PROFILE
    if config.get('fin_mapping_version') != FIN_MAPPING_VERSION:
        print('Warning: model predates the corrected HoloOcean fin mapping; '
              'its policy actions and current evaluation dynamics differ.')
    if args.obstacle_count is not None:
        obstacle_range = (args.obstacle_count, args.obstacle_count)
    else:
        configured_range = config.get('obstacle_count_range', [2, 5])
        obstacle_range = (
            args.min_obstacles if args.min_obstacles is not None else configured_range[0],
            args.max_obstacles if args.max_obstacles is not None else configured_range[1],
        )
    assist_config = config.get('evaluation_assist', {})
    goal_guidance = (args.goal_guidance if args.goal_guidance is not None
                     else assist_config.get(
                         'goal_guidance', config.get('goal_guidance', True)))
    capture_assist = (args.capture_assist if args.capture_assist is not None
                      else assist_config.get(
                          'capture_assist', config.get('capture_assist', False)))
    cruise_assist = (args.cruise_assist if args.cruise_assist is not None
                     else assist_config.get(
                         'cruise_assist', config.get('cruise_assist', True)))
    if obstacle_range[0] < 0 or obstacle_range[1] < obstacle_range[0]:
        parser.error('invalid obstacle range')

    torch.set_num_threads(2)
    env = make_env(args.episode_steps, goal_guidance, vehicle_profile,
                   obstacle_range, capture_assist, cruise_assist,
                   include_path_observation, assist_config)
    if capture_assist or goal_guidance:
        print(f'Evaluation assistance: goal_guidance={goal_guidance}; '
              f'capture_assist={capture_assist}; '
              f'capture_min_throttle={env.capture_min_throttle}; '
              f'capture_turn_distance={env.capture_turn_distance:.1f} m')
    if config.get('dijkstra_reward_version') != env.dijkstra_reward_version:
        print('Warning: saved model used a different Dijkstra reward version; '
              'reported rewards are not comparable with its training run.')
    model = PPO.load(str(args.model), device='cpu')
    fig = ax3d = ax2d = None
    if args.live:
        plt.ion()
        fig = plt.figure(figsize=(15, 6))
        grid = fig.add_gridspec(1, 2, wspace=0.36)
        ax3d = fig.add_subplot(grid[0, 0], projection='3d')
        ax2d = fig.add_subplot(grid[0, 1])
    results = []
    path = info = None
    for episode in range(args.episodes):
        seed = args.seed + episode
        result, path, info, reference_path, grid_bounds = run_episode(
            env, model, seed, live=args.live, render_every=args.render_every,
            figure=(fig, ax3d, ax2d),
        )
        results.append(result)
        print(f'Episode {episode + 1}/{args.episodes}; seed={seed}; '
              f'success={result["success"]}; collision={result["collision"]}; '
              f'out_of_bounds={result["out_of_bounds"]}; '
              f'steps={result["steps"]}; reward={result["reward"]:.2f}; '
              f'final={result["final_distance_m"]:.2f} m; '
              f'depth_error={result["final_depth_error_m"]:.2f} m; '
              f'min={result["min_distance_m"]:.2f} m')

    if not args.live:
        fig = plt.figure(figsize=(15, 6))
        grid = fig.add_gridspec(1, 2, wspace=0.36)
        ax3d = fig.add_subplot(grid[0, 0], projection='3d')
        ax2d = fig.add_subplot(grid[0, 1])
    last = results[-1]
    draw_evaluation(fig, ax3d, ax2d, path, env, last['reward'],
                    last['final_distance_m'], last['steps'], reference_path,
                    grid_bounds)
    args.output.mkdir(parents=True, exist_ok=True)
    png_path = args.output / 'evaluation.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    if args.live or args.show:
        plt.ioff()
        plt.show()
    plt.close(fig)
    csv_path = args.csv or args.output / 'evaluation_metrics.csv'
    (success_rate, collision_rate, out_of_bounds_rate,
     timeout_rate, means) = write_metrics(csv_path, results)
    print(f'Saved {png_path} for the last episode and {csv_path}; '
          f'success_rate={success_rate:.1%}; collision_rate={collision_rate:.1%}; '
          f'out_of_bounds_rate={out_of_bounds_rate:.1%}; '
          f'timeout_rate={timeout_rate:.1%}; mean_reward={means["reward"]:.2f}; '
          f'mean_steps={means["steps"]:.1f}; '
          f'mean_distance={means["final_distance_m"]:.2f} m; '
          f'mean_depth_error={means["final_depth_error_m"]:.2f} m; '
          f'mean_min_distance={means["min_distance_m"]:.2f} m; '
          f'mean_path={means["path_length_m"]:.2f} m')
    env.close()


if __name__ == '__main__':
    main()
