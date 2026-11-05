"""在 HoloOcean 窗口中回放三路声呐 PPO 策略。"""
import argparse
import json
from pathlib import Path

import numpy as np

from highspeed_uuv.holoocean_control import torpedo_fin_commands
from highspeed_uuv.holoocean_env import HoloOceanTorpedoEnv
from highspeed_uuv.policy import BoundedActorCriticPolicy, SonarCNNFeaturesExtractor
from highspeed_uuv.vehicle_profile import (DEFAULT_PROFILE, FIN_MAPPING_VERSION,
                                           actuator_overrides)


ASSIST_ATTRIBUTES = {
    "capture_turn_distance_m": "capture_turn_distance",
    "capture_turn_angle_rad": "capture_turn_angle",
    "capture_min_throttle": "capture_min_throttle",
    "capture_steering_gain": "capture_steering_gain",
    "capture_vertical_gain": "capture_vertical_gain",
    "capture_min_rudder": "capture_min_rudder",
}


def scenario(vehicle_profile, render_sonar=False):
    sensors = [
        {
            "sensor_type": "DynamicsSensor",
            "sensor_name": "DynamicsSensor",
            "configuration": {"UseCOM": True, "UseRPY": False},
        },
        {"sensor_type": "IMUSensor", "sensor_name": "IMUSensor", "Hz": 50},
    ]
    if render_sonar:
        sensors.extend([
            {
                "sensor_type": "ImagingSonar",
                "sensor_name": "FrontSonar",
                "Hz": 5,
                "configuration": {
                    "Azimuth": 120,
                    "Elevation": 30,
                    "RangeMin": 0.5,
                    "RangeMax": 40,
                    "RangeBins": 256,
                    "AzimuthBins": 256,
                },
            },
            {
                "sensor_type": "SidescanSonar",
                "sensor_name": "SidescanSonar",
                "Hz": 5,
                "configuration": {
                    "Azimuth": 120,
                    "Elevation": 5,
                    "RangeMin": 0.5,
                    "RangeMax": 40,
                    "RangeBins": 400,
                },
            },
        ])
    return {
        "name": "highspeed_uuv_policy_visualization",
        "package_name": "Ocean",
        "world": "OpenWater",
        "main_agent": "auv0",
        "ticks_per_sec": 50,
        "agents": [{
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "control_scheme": 1,
            "fossen_model": "torpedo",
            "control_mode": "manualControl",
            "actuator": actuator_overrides(vehicle_profile),
            "location": [0.0, 0.0, -10.0],
            "rotation": [0.0, 0.0, 0.0],
            "sensors": sensors,
        }],
    }


def _reference_environment(config, seed, ticks_per_sec):
    obstacle_range = tuple(config.get("obstacle_count_range", [2, 5]))
    include_path_observation = config.get(
        "include_path_observation",
        config.get("observation_type") == "three_sonar_path_v2",
    )
    env = HoloOceanTorpedoEnv(
        max_steps=config.get("episode_steps", 2500),
        obstacle_count_range=obstacle_range,
        goal_guidance=config.get("goal_guidance", False),
        capture_assist=config.get("capture_assist", False),
        cruise_assist=config.get("cruise_assist", False),
        include_path_observation=include_path_observation,
        easy_rehearsal_probability=0.0,
        ticks_per_sec=ticks_per_sec,
        vehicle_profile=config.get("vehicle_profile", DEFAULT_PROFILE),
    )
    assist = config.get("evaluation_assist", {})
    if "goal_guidance" in assist:
        env.goal_guidance = bool(assist["goal_guidance"])
    if "capture_assist" in assist:
        env.capture_assist = bool(assist["capture_assist"])
    if "cruise_assist" in assist:
        env.cruise_assist = bool(assist["cruise_assist"])
    for key, attribute in ASSIST_ATTRIBUTES.items():
        if key in assist:
            setattr(env, attribute, assist[key])
    env.reset(seed=seed)
    return env


def _apply_configured_assists(env, action, observation):
    action = np.asarray(action, dtype=float).copy()
    if env.goal_guidance:
        action = env._apply_goal_guidance(action, observation)
    if env.cruise_assist:
        action = env._apply_cruise_floor(action, observation)
    if env.capture_assist:
        action = env._keep_turning_near_goal(action, observation)
    return np.clip(action, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="在 HoloOcean 窗口中可视化回放 PPO 策略"
    )
    parser.add_argument("model", type=Path, help="PPO .zip 模型路径")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；应与 evaluate 的 --seed 一致")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="最长回放时间")
    parser.add_argument("--render-sonar", action="store_true",
                        help="在窗口中额外渲染 HoloOcean 前视/侧扫声呐")
    parser.add_argument("--headless", action="store_true",
                        help="不显示 HoloOcean 窗口，仅运行策略和打印状态")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    config_path = args.model.parent / "run.json"
    if not config_path.is_file():
        parser.error(f"run.json not found next to model: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("observation_type") not in (
            "three_sonar_maps_v1", "three_sonar_path_v2"):
        parser.error("model is not a supported three-sonar PPO model")
    if config.get("fin_mapping_version") != FIN_MAPPING_VERSION:
        parser.error("model uses an incompatible fin mapping")

    import holoocean
    from holoocean.fossen_dynamics import FossenInterface
    from holoocean.fossen_dynamics.helper_functions import convert_NWU_to_NED
    from stable_baselines3 import PPO

    vehicle_profile = config.get("vehicle_profile", DEFAULT_PROFILE)
    ticks_per_sec = 50
    world_config = scenario(vehicle_profile, args.render_sonar)
    model = PPO.load(str(args.model), device="cpu")
    reference_env = _reference_environment(config, args.seed, ticks_per_sec)
    if model.observation_space != reference_env.observation_space:
        reference_env.close()
        raise ValueError("saved policy observation space differs from current environment")

    env = holoocean.make(
        scenario_cfg=world_config,
        show_viewport=not args.headless,
        verbose=False,
    )
    try:
        env.set_render_quality(0)
        if not args.headless:
            env.should_render_viewport(True)
        for index, (center, radius) in enumerate(reference_env.obstacles, start=1):
            env.spawn_prop(
                "sphere",
                location=[float(center[0]), -float(center[1]), -float(center[2])],
                scale=float(2.0 * radius),
                sim_physics=False,
                material="gold",
                tag=f"training_obstacle_{index}",
            )
        env.spawn_prop(
            "sphere",
            location=[float(reference_env.goal[0]),
                      -float(reference_env.goal[1]),
                      -float(reference_env.goal[2])],
            scale=float(2.0 * reference_env.goal_radius),
            sim_physics=False,
            material="white",
            tag="training_goal_region",
        )

        print(
            f"Loaded {args.model}; seed={args.seed}; "
            f"obstacles={len(reference_env.obstacles)}; "
            f"goal_radius={reference_env.goal_radius:.1f} m; "
            f"goal_guidance={reference_env.goal_guidance}; "
            f"capture_assist={reference_env.capture_assist}; "
            f"cruise_assist={reference_env.cruise_assist}",
            flush=True,
        )
        print("Gold spheres are obstacles; the white sphere is the goal capture region.",
              flush=True)

        interface = FossenInterface(["auv0"], world_config)
        interface.set_control_mode("auv0", "manualControl")
        accel = np.zeros(6, dtype=float)
        previous_eta = None
        final_eta = None
        terminated_reason = None
        for step in range(int(args.seconds * ticks_per_sec)):
            data = env.step(accel)
            eta, nu = convert_NWU_to_NED(np.asarray(data["DynamicsSensor"], dtype=float))
            final_eta = eta.copy()
            observation = reference_env.observation_from_state(eta, nu)

            if previous_eta is None:
                segment_distance = float(np.linalg.norm(reference_env.goal - eta[:3]))
            else:
                segment_distance = reference_env._segment_distance(
                    reference_env.goal, previous_eta[:3], eta[:3]
                )
            collided = any(
                np.linalg.norm(eta[:3] - center) <= radius
                for center, radius in reference_env.obstacles
            )
            reached = segment_distance <= reference_env.goal_radius
            out_of_bounds = reference_env._outside_planning_bounds(eta[:3])
            if collided or reached or out_of_bounds:
                terminated_reason = (
                    "collision" if collided else "goal reached"
                    if reached else "out of bounds"
                )
                interface.set_u_control("auv0", np.zeros(5, dtype=float))
                print(f"Replay ended: {terminated_reason} at t={step / ticks_per_sec:.2f}s, "
                      f"NED position={eta[:3].round(3)}", flush=True)
                break

            raw_action, _ = model.predict(observation, deterministic=True)
            action = _apply_configured_assists(reference_env, raw_action, observation)
            actuator = interface.vehicles["auv0"].actuator_parameters
            fin_limit = np.deg2rad(actuator["deltaMax_fin_deg"])
            rudder = action[0] * fin_limit
            stern = action[1] * fin_limit
            rpm = (action[2] + 1.0) * 0.5 * actuator["nMax"]
            interface.set_u_control(
                "auv0", torpedo_fin_commands(rudder, stern, rpm)
            )
            accel = interface.update("auv0", data)
            previous_eta = eta.copy()

            if step % ticks_per_sec == 0:
                distance = float(np.linalg.norm(reference_env.goal - eta[:3]))
                print(
                    f"t={step / ticks_per_sec:5.1f}s  "
                    f"distance={distance:6.2f} m  surge={nu[0]:5.2f} m/s  "
                    f"action=[{action[0]: .2f}, {action[1]: .2f}, {action[2]: .2f}]",
                    flush=True,
                )

        if terminated_reason is None and final_eta is not None:
            distance = float(np.linalg.norm(reference_env.goal - final_eta[:3]))
            print(f"Replay time limit reached; final distance={distance:.2f} m")
    finally:
        env.__exit__(None, None, None)
        reference_env.close()


if __name__ == "__main__":
    main()
