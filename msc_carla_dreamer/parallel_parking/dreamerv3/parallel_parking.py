import os
from dreamerv3.carla_parallel_parking_env import CarlaParallelParkingEnv


def make(env=None, **kwargs):
# env (dict) comes from config; kwargs can override
    env = env or {}
    host = env.get("world", {}).get("carla_host", os.getenv("CARLA_HOST", "10.140.13.244"))
    port = env.get("world", {}).get("carla_port", int(os.getenv("CARLA_PORT", "2000")))


    params = {
    "host": host,
    "port": port,
    "town": env.get("world", {}).get("town", "Town10HD_Opt"),
    "image_size": tuple(env.get("obs", {}).get("image_size", (96, 160))),
    "fov": env.get("obs", {}).get("fov", 90),
    "max_steps": env.get("run", {}).get("max_steps", 800),
    "target_width": env.get("task", {}).get("target_width", 2.5),
    "target_length": env.get("task", {}).get("target_length", 6.0),
    "hold_steps": env.get("task", {}).get("hold_steps", 10),
    "success_yaw_tol_deg": env.get("task", {}).get("success_yaw_tol_deg", 10.0),
    "success_pos_tol": env.get("task", {}).get("success_pos_tol", 0.5),
    "seed": env.get("run", {}).get("seed", 0),
    }
    return CarlaParallelParkingEnv(**params)