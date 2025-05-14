import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import (
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)
from torch import nn

from .hyper_layer import HyMlpExtractor

SelfHyBaseModel = TypeVar("SelfHyBaseModel", bound="HyBaseModel")


class HyBaseModel(nn.Module):
    value_optimizer: th.optim.Optimizer
    disc_optimizer: th.optim.Optimizer
    con_optimizer: th.optim.Optimizer

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        # Automatically deactivate dtype and bounds checks
        if normalize_images is False and issubclass(
            features_extractor_class, (NatureCNN, CombinedExtractor)
        ):
            self.features_extractor_kwargs.update(dict(normalized_image=True))

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update(
            dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim,
            )
        )
        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )

    def extract_features(
        self, obs: th.Tensor, features_extractor: BaseFeaturesExtractor
    ) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

         :param obs: The observation
         :param features_extractor: The features extractor to use.
         :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save(
            {
                "state_dict": self.state_dict(),
                "data": self._get_constructor_parameters(),
            },
            path,
        )

    @classmethod
    def load(
        cls: Type[SelfHyBaseModel], path: str, device: Union[th.device, str] = "auto"
    ) -> SelfHyBaseModel:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        th.nn.utils.vector_to_parameters(
            th.as_tensor(vector, dtype=th.float, device=self.device), self.parameters()
        )

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """
        return (
            th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()
        )

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def is_vectorized_observation(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> bool:
        """
        Check whether or not the observation is vectorized,
        apply transposition to image (so that they are channel-first) if needed.
        This is used in DQN when sampling random action (epsilon-greedy policy)

        :param observation: the input observation to check
        :return: whether the given observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                vectorized_env = vectorized_env or is_vectorized_observation(
                    maybe_transpose(obs, obs_space), obs_space
                )
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space),
                self.observation_space,
            )
        return vectorized_env

    def obs_to_tensor(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[th.Tensor, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(
                    obs_, obs_space
                )
                # Add batch dimension if needed
                observation[key] = obs_.reshape(
                    (-1, *self.observation_space[key].shape)
                )

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(
                observation, self.observation_space
            )
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env


class HyBasePolicy(HyBaseModel, ABC):
    features_extractor: BaseFeaturesExtractor

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output
        self.action_space_disc: spaces.Dict = self.action_space["discrete_action"]
        self.action_space_con: spaces.Box = self.action_space["continuous_action"]

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        """"""

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions_disc, actions_con = self._predict(
                observation, deterministic=deterministic
            )
        actions_disc = (
            actions_disc.cpu().numpy().reshape((-1, *self.action_space_disc.shape))
        )
        actions_con = (
            actions_con.cpu().numpy().reshape((-1, *self.action_space_con.shape))
        )
        actions_con = np.clip(
            actions_con, self.action_space_con.low, self.action_space_con.high
        )
        # Remove batch dimension if needed
        if not vectorized_env:
            # actions_disc = actions_disc.squeeze(axis=0)
            actions_con = actions_con.squeeze(axis=0)
            actions = np.concatenate([actions_disc, actions_con], axis=-1)
        else:
            actions = np.concatenate([actions_disc[:, None], actions_con], axis=-1)
        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.action_space_con.low, self.action_space_con.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.action_space_con.low, self.action_space_con.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


class HyActorCriticPolicy(HyBasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        if (
            isinstance(net_arch, list)
            and len(net_arch) > 0
            and isinstance(net_arch[0], dict)
        ):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        if not share_features_extractor:
            raise "share_features_extractor must be True"
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor

        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        self.action_dist_disc = make_proba_distribution(
            self.action_space_disc, dist_kwargs=None
        )
        self.action_dist_con = make_proba_distribution(
            self.action_space_con, use_sde=use_sde, dist_kwargs=dist_kwargs
        )

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HyMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net_disc = self.action_dist_disc.proba_distribution_net(
            latent_dim=latent_dim_pi
        )
        self.action_net_con, self.log_std = self.action_dist_con.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net_con: 0.01,
                self.action_net_disc: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        value_parameters = [
            self.value_net.parameters(),
            self.mlp_extractor.value_net.parameters(),
            self.features_extractor.parameters(),
        ]
        self.value_parameters = [p for group in value_parameters for p in group]

        disc_parameters = [
            self.action_net_disc.parameters(),
            self.mlp_extractor.policy_net_disc.parameters(),
        ]
        self.disc_parameters = [p for group in disc_parameters for p in group]

        con_parameters = [
            self.action_net_con.parameters(),
            [self.log_std],
            self.mlp_extractor.policy_net_con.parameters(),
        ]
        self.con_parameters = [
            p
            for group in con_parameters
            for p in (group if isinstance(group, list) else list(group))
        ]

        self.value_optimizer = self.optimizer_class(
            self.value_parameters, lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.disc_optimizer = self.optimizer_class(
            self.disc_parameters, lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.con_optimizer = self.optimizer_class(
            self.con_parameters, lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)

        latent_pi_disc, latent_pi_con, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        distribution_disc = self._get_action_dist_from_latent_disc(latent_pi_disc)
        actions_disc = distribution_disc.get_actions(deterministic=deterministic)
        log_prob_disc = distribution_disc.log_prob(actions_disc)

        distribution_con = self._get_action_dist_from_latent_con(latent_pi_con)
        actions_con = distribution_con.get_actions(deterministic=deterministic)
        log_prob_con = distribution_con.log_prob(actions_con)

        actions_disc = actions_disc.reshape((-1, *self.action_space_disc.shape))
        actions_con = actions_con.reshape((-1, *self.action_space_con.shape))
        return actions_disc, actions_con, values, log_prob_disc, log_prob_con

    def extract_features(
        self, obs: th.Tensor
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        return super().extract_features(obs, self.features_extractor)

    def _get_action_dist_from_latent_disc(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = self.action_net_disc(latent_pi)
        return self.action_dist_disc.proba_distribution(action_logits=mean_actions)

    def _get_action_dist_from_latent_con(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = self.action_net_con(latent_pi)
        if isinstance(self.action_dist_con, DiagGaussianDistribution):
            return self.action_dist_con.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist_con, StateDependentNoiseDistribution):
            return self.action_dist_con.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        distribution_disc, distribution_con = self.get_distribution(observation)
        return distribution_disc.get_actions(
            deterministic=deterministic
        ), distribution_con.get_actions(deterministic=deterministic)

    def evaluate_actions(
        self, obs: th.Tensor, actions_disc: th.Tensor, actions_con: th.Tensor
    ) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]
    ]:
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        detached_f = features.detach()
        latent_pi_disc = self.mlp_extractor.forward_actor_disc(detached_f)
        latent_pi_con = self.mlp_extractor.forward_actor_con(detached_f)
        distribution_disc = self._get_action_dist_from_latent_disc(latent_pi_disc)
        log_prob_disc = distribution_disc.log_prob(actions_disc)
        entropy_disc = distribution_disc.entropy()
        distribution_con = self._get_action_dist_from_latent_con(latent_pi_con)
        log_prob_con = distribution_con.log_prob(actions_con)
        entropy_con = distribution_con.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob_disc, log_prob_con, entropy_disc, entropy_con

    def get_distribution(self, obs: th.Tensor) -> Tuple[Distribution, Distribution]:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi_disc = self.mlp_extractor.forward_actor_disc(features)
        latent_pi_con = self.mlp_extractor.forward_actor_con(features)
        return self._get_action_dist_from_latent_disc(
            latent_pi_disc
        ), self._get_action_dist_from_latent_con(latent_pi_con)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def reset_noise(self, n_envs: int = 1) -> None:
        assert isinstance(self.action_dist_con, StateDependentNoiseDistribution), (
            "reset_noise() is only available when using gSDE"
        )
        self.action_dist_con.sample_weights(self.log_std, batch_size=n_envs)


class HyActorCriticCnnPolicy(HyActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class HyMultiInputActorCriticPolicy(HyActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
