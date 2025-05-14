import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from .hy_policies import (
    HyActorCriticCnnPolicy,
    HyActorCriticPolicy,
    HyBasePolicy,
    HyMultiInputActorCriticPolicy,
)
from .hy_on_policy_algo import HyOnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfHyPPO = TypeVar("SelfHyPPO", bound="HyPPO")


class HyPPO(HyOnPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[HyBasePolicy]]] = {
        "MlpPolicy": HyActorCriticPolicy,
        "CnnPolicy": HyActorCriticCnnPolicy,
        "MultiInputPolicy": HyMultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[HyActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef_con: float = 0.0,
        ent_coef_disc: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef_con=ent_coef_con,
            ent_coef_disc=ent_coef_disc,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Dict),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert batch_size > 1, (
                "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
            )

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (not normalize_advantage), (
                f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        self._update_learning_rate(self.policy.value_optimizer)
        self._update_learning_rate(self.policy.disc_optimizer)
        self._update_learning_rate(self.policy.con_optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses_disc, entropy_losses_con = [], []
        pg_losses_disc, pg_losses_con, value_losses = [], [], []
        clip_fractions_disc, clip_fractions_con = [], []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs_con = []
            approx_kl_divs_disc = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions_disc = rollout_data.actions_disc.long().flatten()
                actions_con = rollout_data.actions_con

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob_disc, log_prob_con, entropy_disc, entropy_con = (
                    self.policy.evaluate_actions(
                        rollout_data.observations, actions_disc, actions_con
                    )
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio_disc = th.exp(log_prob_disc - rollout_data.old_log_probs_disc)
                ratio_con = th.exp(log_prob_con - rollout_data.old_log_probs_con)

                # clipped surrogate loss
                policy_loss_1_disc = advantages * ratio_disc
                policy_loss_2_disc = advantages * th.clamp(
                    ratio_disc, 1 - clip_range, 1 + clip_range
                )
                policy_loss_disc = -th.min(
                    policy_loss_1_disc, policy_loss_2_disc
                ).mean()

                policy_loss_1_con = advantages * ratio_con
                policy_loss_2_con = advantages * th.clamp(
                    ratio_con, 1 - clip_range, 1 + clip_range
                )
                policy_loss_con = -th.min(policy_loss_1_con, policy_loss_2_con).mean()

                # Logging
                pg_losses_disc.append(policy_loss_disc.item())
                clip_fraction_disc = th.mean(
                    (th.abs(ratio_disc - 1) > clip_range).float()
                ).item()
                clip_fractions_disc.append(clip_fraction_disc)

                # Logging
                pg_losses_con.append(policy_loss_con.item())
                clip_fraction_con = th.mean(
                    (th.abs(ratio_con - 1) > clip_range).float()
                ).item()
                clip_fractions_con.append(clip_fraction_con)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy_disc is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_disc = -th.mean(-log_prob_disc)
                else:
                    entropy_loss_disc = -th.mean(entropy_disc)

                if entropy_con is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_con = -th.mean(-log_prob_con)
                else:
                    entropy_loss_con = -th.mean(entropy_con)

                entropy_losses_disc.append(entropy_loss_disc.item())
                entropy_losses_con.append(entropy_loss_con.item())

                # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss_disc = policy_loss_disc + self.ent_coef_disc * entropy_loss_disc
                self.policy.disc_optimizer.zero_grad()
                loss_disc.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.disc_parameters, self.max_grad_norm
                )
                self.policy.disc_optimizer.step()
                loss_con = policy_loss_con + self.ent_coef_con * entropy_loss_con
                self.policy.con_optimizer.zero_grad()
                loss_con.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.con_parameters, self.max_grad_norm
                )
                self.policy.con_optimizer.step()

                loss_value = self.vf_coef * value_loss
                self.policy.value_optimizer.zero_grad()
                loss_value.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.value_parameters, self.max_grad_norm
                )
                self.policy.value_optimizer.step()

                with th.no_grad():
                    log_ratio_disc = log_prob_disc - rollout_data.old_log_probs_disc
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio_disc) - 1) - log_ratio_disc)
                        .cpu()
                        .numpy()
                    )
                    approx_kl_divs_disc.append(approx_kl_div)

                    log_ratio_con = log_prob_disc - rollout_data.old_log_probs_disc
                    approx_kl_div_con = (
                        th.mean((th.exp(log_ratio_con) - 1) - log_ratio_con)
                        .cpu()
                        .numpy()
                    )
                    approx_kl_divs_con.append(approx_kl_div_con)

                # self.policy.optimizer.zero_grad()
                # loss.backward()
                # # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss_disc", np.mean(entropy_losses_disc))
        self.logger.record("train/entropy_loss_con", np.mean(entropy_losses_con))
        self.logger.record("train/policy_gradient_loss_disc", np.mean(pg_losses_disc))
        self.logger.record("train/policy_gradient_loss_con", np.mean(pg_losses_con))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl_disc", np.mean(approx_kl_divs_disc))
        self.logger.record("train/approx_kl_con", np.mean(approx_kl_divs_con))
        self.logger.record("train/clip_fraction_disc", np.mean(clip_fractions_disc))
        self.logger.record("train/clip_fraction_con", np.mean(clip_fractions_con))
        self.logger.record("train/loss_disc", loss_disc.item())
        self.logger.record("train/loss_con", loss_con.item())
        self.logger.record("train/loss_value", loss_value.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfHyPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHyPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
