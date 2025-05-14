from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import tyro
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from hppo.hy_ppo import HyPPO
from wrapper import SB3HyPPOWrapper


@dataclass
class Args:
    env: str
    n_envs: int = 8
    seed: int = 42
    save_path: Path = Path(f"output/sb3hppo_{datetime.now():%Y%m%d_%H%M%S}")
    total_timesteps: int = 1000_0000


args = tyro.cli(Args)
env = make_vec_env(
    lambda: SB3HyPPOWrapper(gym.make(f"gymnasium_hybrid:{args.env}")),
    n_envs=args.n_envs,
)
algo = HyPPO("MlpPolicy", env, tensorboard_log=args.save_path / "tb_log", seed=args.seed)
checkpoint_callback = CheckpointCallback(
    save_freq=args.total_timesteps // 10,
    save_path=args.save_path / "checkpoints",
    name_prefix="model",
)
algo.learn(
    total_timesteps=args.total_timesteps,
    progress_bar=True,
    log_interval=1,
    callback=checkpoint_callback,
)
algo.save(args.save_path / "model")
