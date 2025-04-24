import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from redexp.utils import normalize_angle
from redexp.config.turtlebot import OBSTACLE_RADIUS
from redexp.robots.turtlebot import Turtlebot


class TurtlebotEnv(gym.Env):
    def __init__(self, goal_conditioned=False, model_mismatch=False):
        self.goal_conditioned = goal_conditioned

        self.goal_location = np.array([1.75, 1.25], dtype=np.float32)
        self.goal_r = 0.15

        self.turtlebot = Turtlebot(
            goal_location=self.goal_location,
            goal_r=self.goal_r,
            model_mismatch=model_mismatch,
        )
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=np.array([-self.turtlebot.dyn.wMax], dtype=np.float32),
            high=np.array([self.turtlebot.dyn.wMax], dtype=np.float32),
            dtype=np.float32,
        )

        if self.goal_conditioned:
            self.observation_space = gym.spaces.Box(
                low=np.array([-6, -6, -1.0, -1.0, -6, -6], dtype=np.float32),
                high=np.array([6, 6, 1.0, 1.0, 6, 6], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([-6, -6, -1.0, -1.0], dtype=np.float32),
                high=np.array([6, 6, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.obstacle_location = np.array([0.0, 0.0], dtype=np.float32)
        self.obstacle_r = OBSTACLE_RADIUS

        self.goal_idx = 0

    def reset(self, seed=None, options={}):
        print("ROBOT NEEDS TO BE RESET")
        input("Press Enter to continue...")

        obs = self._get_obs()

        if self.goal_conditioned:
            # set new goal location
            goal_locations = np.array([
                [0.0, -1.9],
                [0.0, 2.4],
            ])
            self.goal_idx = (1 + self.goal_idx) % 2
            self.goal_location = goal_locations[self.goal_idx]
            print(f"New goal location: {self.goal_location}")
            self.turtlebot.goal_location = self.goal_location
            obs = np.append(obs, self.goal_location)
        
        self.traj_min_brt_value = self.turtlebot.get_brt_value()

        return obs, {"cost": 0, "traj_min_brt_value": self.traj_min_brt_value}

    def step(self, action):
        # print("action value: ",action)
        self.turtlebot.set_action(action)
        state = self.turtlebot.get_state()

        info = {}

        obs = self._get_obs()
        reward = -np.linalg.norm(state[:2] - self.goal_location)
        cost = (
            np.linalg.norm(state[:2] - self.obstacle_location)
            < (self.obstacle_r + self.turtlebot.dyn.r)
            or not self.turtlebot.in_bounds()
        )
        terminated = not self.turtlebot.in_bounds()

        terminated = terminated or self.turtlebot.reach_goal()
        if self.turtlebot.reach_goal():
            info['reach_goal'] = True

        terminated = terminated or cost

        info["cost"] = cost
        brt_value = self.turtlebot.get_brt_value()
        self.traj_min_brt_value = min(brt_value, self.traj_min_brt_value)

        info['traj_min_brt_value'] = self.traj_min_brt_value

        if self.goal_conditioned:
            obs = np.append(obs, self.goal_location)

        return obs, reward, terminated, False, info

    def _get_obs(self):
        self.state = self.turtlebot.get_state()

        return np.array(
            [self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2])]
        )
