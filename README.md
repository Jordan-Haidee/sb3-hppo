# SB3HPPO

> **Hybrid PPO** implementation using stable-baselines3 and benchmarked in the Gymnasium-Hybrid standard environment.

## Usage
### 1. Clone this repo
```shell
git clone ...
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

╭─ options ───────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit             │
│ --env STR               (required)                                  │
│ --save-path PATH        (default: sb3ppo_{%Y%m%d_%H%M%S})           │
│ --n-envs INT            (default: 8)                                │
│ --total-timesteps INT   (default: 10000000)                         │
╰─────────────────────────────────────────────────────────────────────╯
```
Example:

```shell
python train.py --env Moving-v0 # or Sliding-v0, HardMove-v0
```
The trained policy and tensorboard log will be saved at `output/sb3hppo_xxx/model.zip` and `output/sb3hppo_xxx/tb_log`, respectively.

### 4. Test your policy
```
$ python test.py --help
usage: test.py [-h] --env STR --ckpt PATH [--render | --no-render]

╭─ options ───────────────────────────────────────────────╮
│ -h, --help              show this help message and exit │
│ --env STR               (required)                      │
│ --ckpt PATH             (required)                      │
│ --render, --no-render   (default: False)                │
╰─────────────────────────────────────────────────────────╯
```
Example:
```shell
python test.py --env "Moving-v0" --ckpt output/sb3hppo_20250514_210451/model.zip --render
```
## Acknowledgement
Thanks to [@wild-firefox](https://github.com/wild-firefox) and [@CAI23sbP](https://github.com/CAI23sbP) ! This repo heavily depends on their preceding works:

- [Gymnasium-Hybrid](https://github.com/wild-firefox/gymnasium_hybrid)
- [Hybrid-Action-PPO](https://github.com/CAI23sbP/Hybrid-Action-PPO)
