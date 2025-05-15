import json
from dataclasses import asdict, dataclass
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
    env: str = "Moving-v0"  # env id from gymnasium_hybrid (Moving-v0 / Sliding-v0 / HardMove-v0)
    n_envs: int = 8  # number of parallel environments
    seed: int = 42  # random seed
    save_path: Path | None = None  # path to save model and logs
    total_timesteps: int = 500_0000  # total timesteps to train

    def __post_init__(self):
        if self.save_path is None:
            self.save_path = Path(
                f"output/sb3hppo_{self.env}_{datetime.now():%Y%m%d_%H%M%S}"
            )

    def asdict(self):
        data = asdict(self)
        data.update({"save_path": str(data["save_path"])})
        return data


# parse command line arguments
args = tyro.cli(Args)

# save args
args.save_path.mkdir(parents=True, exist_ok=True)
with open(args.save_path / "args.json", "w") as f:
    json.dump(args.asdict(), f, indent=4)

# build parallel env
env = make_vec_env(
    lambda: SB3HyPPOWrapper(gym.make(f"gymnasium_hybrid:{args.env}")),
    n_envs=args.n_envs,
)

# init algo
algo = HyPPO(
    "MlpPolicy", env, tensorboard_log=args.save_path / "tb_log", seed=args.seed
)

# set save callback
checkpoint_callback = CheckpointCallback(
    save_freq=args.total_timesteps // args.n_envs // 10,
    save_path=args.save_path / "checkpoints",
    name_prefix="model",
)

# training
algo.learn(
    total_timesteps=args.total_timesteps,
    progress_bar=True,
    log_interval=1,
    callback=checkpoint_callback,
)

# save final model
algo.save(args.save_path / "model")
