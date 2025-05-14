import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from typing import NamedTuple, Optional, Union, Generator
from stable_baselines3.common.buffers import BaseBuffer


def get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), (
            "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        )
        return int(action_space.n)
    elif isinstance(action_space, spaces.Dict):
        return int(np.prod(action_space["continuous_action"].shape)), 1
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


class HYRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions_con: th.Tensor
    actions_disc: th.Tensor
    old_values: th.Tensor
    old_log_probs_con: th.Tensor
    old_log_probs_disc: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class HYRolloutBuffer(BaseBuffer):
    observations: np.ndarray
    actions_con: np.ndarray
    actions_disc: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs_con: np.ndarray
    log_probs_disc: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_con_dim, self.action_disc_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32
        )
        self.actions_disc = np.zeros(
            (self.buffer_size, self.n_envs, self.action_disc_dim), dtype=np.float32
        )
        self.actions_con = np.zeros(
            (self.buffer_size, self.n_envs, self.action_con_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs_disc = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.log_probs_con = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action_disc: np.ndarray,
        action_con: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_probs_disc: th.Tensor,
        log_probs_con: th.Tensor,
    ):
        self.observations[self.pos] = np.array(obs).copy()
        self.actions_disc[self.pos] = np.array(action_disc).copy()
        self.actions_con[self.pos] = np.array(action_con).copy()
        self.log_probs_disc[self.pos] = log_probs_disc.clone().cpu().numpy()
        self.log_probs_con[self.pos] = log_probs_con.clone().cpu().numpy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def filter(self):
        obs = self.observations[: self.pos]
        return np.mean(obs), np.std(obs)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[HYRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions_con",
                "actions_disc",
                "values",
                "log_probs_con",
                "log_probs_disc",
                "advantages",
                "returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> HYRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.actions_con[batch_inds],
            self.actions_disc[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs_con[batch_inds].flatten(),
            self.log_probs_disc[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return HYRolloutBufferSamples(*tuple(map(self.to_torch, data)))
