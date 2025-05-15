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
    save_video: Path | None = None  # Path to save the video (None to disable)

    def __post_init__(self):
        if self.render:
            self.save_video = None


args = tyro.cli(Args)
if args.render:
    render_mode = "human"
elif args.save_video is not None:
    render_mode = "rgb_array"
else:
    render_mode = None

env = gym.make(f"gymnasium_hybrid:{args.env}", render_mode=render_mode)
env = SB3HyPPOWrapper(env)
if args.render is False and args.save_video is not None:
    env = gym.wrappers.RecordVideo(
        env,
        args.save_video,
        name_prefix=args.env,
    )

algo = HyPPO.load(args.ckpt, custom_objects={"tensorboard_log": None})

total_r = 0
s, _ = env.reset()
while True:
    a, _ = algo.predict(s)
    s, r, t1, t2, _ = env.step(a)
    total_r += r
    if args.render:
        env.render()
    if t1 or t2:
        break
env.close()

print(f"Total reward: {total_r}")
