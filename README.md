# SB3-HPPO

> **Hybrid PPO** implementation using stable-baselines3 and benchmarked in the Gymnasium-Hybrid standard environment.

## Usage
### 1. Clone this repo
```shell
git clone https://github.com/Jordan-Haidee/sb3-hppo.git
cd path/to/sb3ppo
```

### 2. Install dependencies
```shell
uv sync # recommended
# or `pip install requirements.txt`
```

### 3. Run training
```shell
$ python train.py --help
usage: train.py [-h] [OPTIONS]

╭─ options ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                          │
│ --env STR               env id from gymnasium_hybrid (Moving-v0 / Sliding-v0 / HardMove-v0) (default: Moving-v0) │
│ --n-envs INT            number of parallel environments (default: 8)                                             │
│ --seed INT              random seed (default: 42)                                                                │
│ --save-path {None}|PATH                                                                                          │
│                         path to save model and logs (default: None)                                              │
│ --total-timesteps INT   total timesteps to train (default: 5000000)                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
Example:

```shell
python train.py --env Moving-v0 # or Sliding-v0, HardMove-v0
```
The trained policy and tensorboard log will be saved at `output/sb3hppo_xxx/model.zip` and `output/sb3hppo_xxx/tb_log`, respectively.

### 4. Test your policy
```
$ python .\test.py --help
usage: test.py [-h] [OPTIONS]

╭─ options ──────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                │
│ --env STR               Env id from gymnasium_hybrid (Moving-v0 / Sliding-v0 / HardMove-v0) (required) │
│ --ckpt PATH             Path to the checkpoint file (*.zip) (required)                                 │
│ --render, --no-render   Whether to render the environment (default: False)                             │
│ --save-video {None}|PATH                                                                               │
│                         Path to save the video (None to disable) (default: None)                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
Example:
```shell
python test.py --env "Moving-v0" --ckpt output/sb3hppo_Moving-v0_20250515_114301/model.zip --render
```
## Acknowledgement
Thanks to [@wild-firefox](https://github.com/wild-firefox) and [@CAI23sbP](https://github.com/CAI23sbP) ! This repo heavily depends on their preceding works:

- [Gymnasium-Hybrid](https://github.com/wild-firefox/gymnasium_hybrid)
- [Hybrid-Action-PPO](https://github.com/CAI23sbP/Hybrid-Action-PPO)
