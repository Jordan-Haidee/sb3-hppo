from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import tyro

from hppo.hy_ppo import HyPPO
from wrapper import SB3HyPPOWrapper


@dataclass
class Args:
    env: str  # Env id from gymnasium_hybrid (Moving-v0 / Sliding-v0 / HardMove-v0)
    ckpt: Path  # Path to the checkpoint file (*.zip)
    render: bool = False  # Whether to render the environment


args = tyro.cli(Args)
env = SB3HyPPOWrapper(
    gym.make(
        f"gymnasium_hybrid:{args.env}", render_mode="human" if args.render else None
    )
)
algo = HyPPO.load(args.ckpt)
s, _ = env.reset()
while True:
    a, _ = algo.predict(s)
    s, r, t1, t2, _ = env.step(a)
    if args.render:
        env.render()
    if t1 or t2:
        break
