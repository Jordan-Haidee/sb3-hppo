from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import tyro
from stable_baselines3.common.env_util import make_vec_env

from hppo.hy_ppo import HyPPO
from wrapper import SB3HyPPOWrapper


@dataclass
class Args:
    env: str
    save_path: Path = Path(f"output/sb3hppo_{datetime.now():%Y%m%d_%H%M%S}")
    n_envs: int = 8
    total_timesteps: int = 1000_0000


args = tyro.cli(Args)
env = make_vec_env(
    lambda: SB3HyPPOWrapper(gym.make(f"gymnasium_hybrid:{args.env}")),
    n_envs=args.n_envs,
)
algo = HyPPO("MlpPolicy", env, tensorboard_log=args.save_path / "tb_log")
algo.learn(total_timesteps=args.total_timesteps, progress_bar=True, log_interval=1)
algo.save(args.save_path / "model")
