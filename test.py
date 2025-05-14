from dataclasses import dataclass

import gymnasium as gym
import tyro

from hppo.hy_ppo import HyPPO
from wrapper import SB3HyPPOWrapper


@dataclass
class Args:
    env: str
    ckpt: str
    render: bool = False


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
    env.render()
    if t1 or t2:
        break
