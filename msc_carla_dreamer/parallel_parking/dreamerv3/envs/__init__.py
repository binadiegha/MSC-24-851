# envs/__init__.py
from gym.envs.registration import register

# Gym ID used by car_dreamer/tasks.yaml
register(
    id="CarlaParallelParking-v0",
    entry_point="envs.carla_parallel_parking_env:make_env",
)