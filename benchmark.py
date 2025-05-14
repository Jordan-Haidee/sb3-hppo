import subprocess
import sys

for s in range(1, 3 + 1):
    for e in ["Moving-v0", "Sliding-v0", "HardMove-v0"]:
        cmd = f"{sys.executable} train.py --env {e} --seed {s}"
        subprocess.run(cmd.split(" "))
