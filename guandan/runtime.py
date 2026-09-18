from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RULESET = "botzone-round-lab-v2"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请选择 cpu / mps 或安装对应的 PyTorch。")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前进程无法使用 MPS，请选择 cpu。")
    if name != "cpu" and name != "mps" and not name.startswith("cuda"):
        raise ValueError(f"未知设备: {name}")
    return name


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def system_info():
    import psutil
    return dict(python=platform.python_version(), platform=platform.platform(),
                cpus=os.cpu_count(), memory_gb=round(psutil.virtual_memory().total / 2**30, 1),
                torch=torch.__version__, cuda=torch.cuda.is_available(),
                mps=torch.backends.mps.is_available(), default_device=device_for(),
                ruleset=RULESET)
