"""Abstract base classes for RL algorithms."""

import io
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import (
    check_for_nested_spaces,
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    recursive_getattr,
    recursive_setattr,
    save_to_zip_file,
)
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    TensorDict,
)
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from stable_baselines3.common.base_class import maybe_make_env

SelfHyBaseAlgorithm = TypeVar("SelfHyBaseAlgorithm", bound="HyBaseAlgorithm")


class HyBaseAlgorithm(ABC):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {}
    policy: BasePolicy
    observation_space: spaces.Space
    action_space: spaces.Space
    n_envs: int
    lr_schedule: Schedule
    _logger: Logger

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str, None],
        learning_rate: Union[float, Schedule],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ) -> None:
        if isinstance(policy, str):
            self.policy_class = self._get_policy_from_name(policy)
        else:
            self.policy_class = policy

        self.device = get_device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise: Optional[ActionNoise] = None
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env: Optional[VecEnv] = None
        self._vec_normalize_env: Optional[VecNormalize] = None
        # Create and wrap the env if needed
        if env is not None:
            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose, monitor_wrapper)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env
            self.action_space_con = self.action_space["continuous_action"]
            self.action_space_disc = self.action_space["discrete_action"]

            # get VecNormalize object if needed
            self._vec_normalize_env = unwrap_vec_normalize(env)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires "
                    "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(
                self.observation_space, spaces.Dict
            ):
                raise ValueError(
                    f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}"
                )

            if self.use_sde and not isinstance(self.action_space_con, spaces.Box):
                raise ValueError(
                    "generalized State-Dependent Exploration (gSDE) can only be used with continuous actions."
                )

            if isinstance(self.action_space_con, spaces.Box):
                assert np.all(
                    np.isfinite(
                        np.array(
                            [self.action_space_con.low, self.action_space_con.high]
                        )
                    )
                ), "Continuous action space must have a finite lower and upper bound"

    @staticmethod
    def _wrap_env(
        env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True
    ) -> VecEnv:
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space)
                        and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(
                    env.observation_space
                ) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def set_logger(self, logger: Logger) -> None:
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ) -> None:
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(
            total_timesteps
        )

    def _update_learning_rate(
        self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        self.logger.record(
            "train/learning_rate", self.lr_schedule(self._current_progress_remaining)
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )

    def _excluded_save_params(self) -> List[str]:
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _get_policy_from_name(self, policy_name: str) -> Type[BasePolicy]:
        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy"]

        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        if isinstance(callback, list):
            callback = CallbackList(callback)

        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if not self._custom_logger:
            self._logger = utils.configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )

        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        return self._vec_normalize_env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        env = self._wrap_env(env, self.verbose)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        self._vec_normalize_env = unwrap_vec_normalize(env)
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    @abstractmethod
    def learn(
        self: SelfHyBaseAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHyBaseAlgorithm:
        """ """

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """ """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """ """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """ """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfHyBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfHyBaseAlgorithm:
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if (
                "net_arch" in data["policy_kwargs"]
                and len(data["policy_kwargs"]["net_arch"]) > 0
            ):
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(
                    saved_net_arch[0], dict
                ):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments"
            )

        # Gym -> Gymnasium space conversion

        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(
                data[key]
            )  # pytype: disable=unsupported-operands

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # pytype: disable=not-instantiable,wrong-keyword-args
        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )
        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]  # pytype: disable=attribute-error
        return model

    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        data = self.__dict__.copy()

        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            var_name = torch_var.split(".")[0]
            exclude.add(var_name)

        for param_name in exclude:
            data.pop(param_name, None)

        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(
            path, data=data, params=params_to_save, pytorch_variables=pytorch_variables
        )
