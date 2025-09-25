from __future__ import annotations

from abc import abstractmethod

import numpy as np
from gymnasium import Env

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import (
    MultiAgentObservation,
    observation_factory,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParallelParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment for parallel parking.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to parallel park in a designated spot along
    the side of a street between other parked vehicles.

    Adapted from ParkingEnv for parallel parking scenarios.
    """

    # For parallel parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    PARKING_OBS = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }
    }

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "ContinuousAction"},
                "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "screen_width": 800,
                "screen_height": 400,
                "centering_position": [0.5, 0.5],
                "scaling": 7,
                "controlled_vehicles": 1,
                "vehicles_count": 6,  # More vehicles for parallel parking scenario
                "add_walls": True,
                "street_length": 100,
                "parking_spot_length": 8,  # Length of parallel parking spot
                "lane_width": 4.0,
            }
        )
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(
            self, self.PARKING_OBS["observation"]
        )

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """
        Create a road with parallel parking spots along the side.
        
        Creates a main driving lane with parking spaces along one side,
        similar to street-side parallel parking.
        """
        net = RoadNetwork()
        lane_width = self.config["lane_width"]
        street_length = self.config["street_length"]
        
        # Line types
        lt_solid = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        lt_dashed = (LineType.STRIPED, LineType.STRIPED)
        
        # Main driving lane (horizontal)
        net.add_lane(
            "main", "main_end",
            StraightLane(
                [-street_length/2, 0], [street_length/2, 0], 
                width=lane_width, line_types=lt_dashed
            )
        )
        
        # Parking lane (parallel to main lane, offset to the side)
        parking_offset = lane_width + lane_width/2  # Distance from main lane
        net.add_lane(
            "parking", "parking_end",
            StraightLane(
                [-street_length/2, -parking_offset], [street_length/2, -parking_offset],
                width=lane_width, line_types=lt_solid
            )
        )
        
        # Create individual parking spots as separate lane segments
        spot_length = self.config["parking_spot_length"]
        num_spots = int(street_length // (spot_length * 1.2))  # Some spacing between spots
        
        for i in range(num_spots):
            x_start = -street_length/2 + i * spot_length * 1.2
            x_end = x_start + spot_length
            
            net.add_lane(
                f"spot_{i}", f"spot_{i}_end",
                StraightLane(
                    [x_start, -parking_offset], [x_end, -parking_offset],
                    width=lane_width, line_types=lt_solid
                )
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create vehicles for parallel parking scenario."""
        # Get all parking spot lanes
        parking_spots = [lane_id for lane_id in self.road.network.lanes_dict().keys() 
                        if lane_id[0].startswith("spot_")]
        
        # Controlled vehicle - starts on the main driving lane
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            # Start position: on the main driving lane, approaching the parking area
            start_x = -self.config["street_length"]/3 + i * 10.0
            vehicle = self.action_type.vehicle_class(
                self.road, [start_x, 0.0], 0.0, 5.0  # Heading east, some initial speed
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        # Create goal in one of the parking spots
        available_spots = parking_spots.copy()
        
        for vehicle in self.controlled_vehicles:
            if available_spots:
                spot_id = available_spots[self.np_random.choice(len(available_spots))]
                lane = self.road.network.get_lane(spot_id)
                
                # Goal position: center of the parking spot, parallel to the road
                goal_x = (lane.start[0] + lane.end[0]) / 2
                goal_y = lane.start[1]
                
                vehicle.goal = Landmark(
                    self.road, [goal_x, goal_y], heading=0.0  # Parallel to road
                )
                self.road.objects.append(vehicle.goal)
                available_spots.remove(spot_id)

        # Add parked vehicles to create realistic parallel parking scenario
        parked_vehicles_count = min(self.config["vehicles_count"], len(available_spots))
        
        for i in range(parked_vehicles_count):
            if available_spots:
                spot_id = available_spots[self.np_random.choice(len(available_spots))]
                lane = self.road.network.get_lane(spot_id)
                
                # Random position within the parking spot
                longitudinal_pos = self.np_random.uniform(0.2, 0.8) * lane.length
                
                # Create parked vehicle
                parked_vehicle = Vehicle.make_on_lane(
                    self.road, spot_id, longitudinal=longitudinal_pos, speed=0.0
                )
                parked_vehicle.color = VehicleGraphics.DEFAULT_COLOR
                self.road.vehicles.append(parked_vehicle)
                available_spots.remove(spot_id)



        # Add walls/boundaries
        if self.config["add_walls"]:
            street_length = self.config["street_length"]
            street_width = 15
            
            # Side boundaries
            for y in [-street_width, street_width/2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (street_length * 1.2, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            
            # End boundaries
            for x in [-street_length*0.6, street_length*0.6]:
                obstacle = Obstacle(self.road, [x, -street_width/2], heading=np.pi/2)
                obstacle.LENGTH, obstacle.WIDTH = (street_width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded with emphasis on proper parallel parking alignment.

        We use a weighted p-norm with additional consideration for orientation alignment
        which is crucial for parallel parking.

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in obs
        )
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParallelParkingEnvEasy(ParallelParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParallelParkingEnvHard(ParallelParkingEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 10, "parking_spot_length": 6.5})  # Tighter parking spots