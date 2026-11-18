"""Record the software/hardware environment and git revision to results/env.json."""
import json
import os
import platform
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(out="results/env.json"):
    import torch
    info = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": run("git rev-parse HEAD"), "git_dirty": bool(run("git status --porcelain")),
        "pip_freeze": (run(f"{sys.executable} -m pip freeze") or "").splitlines(),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=1)
    print({k: v for k, v in info.items() if k != "pip_freeze"})


if __name__ == "__main__":
    main(*sys.argv[1:])
