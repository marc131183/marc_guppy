import numpy as np
import math
import sys

from stable_baselines import DQN

from scipy.spatial import cKDTree

sys.path.append("gym_guppy")
from gym_guppy import BaseCouzinGuppy, TurnBoostAgent
from gym_guppy.tools.math import ray_casting_walls, compute_dist_bins


class MarcGuppy(BaseCouzinGuppy, TurnBoostAgent):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        degrees=360
        num_bins=36 * 2
        self.world_bounds = [np.array([-0.5, -0.5]), np.array([0.5, 0.5])]
        self._model = DQN.load(model_path)

        #has to align with model
        max_turn_rate = np.pi/2
        num_bins_turn_rate = 20
        self._turn_rate_bins = np.linspace(-max_turn_rate, max_turn_rate, num_bins_turn_rate)
        num_bins_speed = 10
        max_speed = 0.10
        self._speed_bins = np.linspace(0.03, max_speed, num_bins_speed)

        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.sector_bounds = np.linspace(-self.cutoff, self.cutoff, num_bins + 1)
        self.ray_directions = np.linspace(-self.cutoff, self.cutoff, num_bins)
        # TODO: is this faster than just recreating the array?
        self.obs_placeholder = np.empty((2, num_bins))


    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        self.obs_placeholder[0] = compute_dist_bins(state[0], state[1:], self.sector_bounds, self.diagonal * 1.1)
        self.obs_placeholder[1] = ray_casting_walls(state[0], self.world_bounds, self.ray_directions, self.diagonal * 1.1)

        action, _ = self._model.predict(self.obs_placeholder)

        turn_rate = math.floor(action/len(self._speed_bins))
        speed = action%len(self._speed_bins)
        turn, speed = self._turn_rate_bins[turn_rate], self._speed_bins[speed]

        self.turn = turn
        self.speed = speed