from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import os
import platform
import random
import subprocess
import numpy as np
import torch


def environment():
    packages = {}
    for name in ("torch", "numpy", "rasterio", "matplotlib", "pytest", "affine"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "packages": packages, "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = [{"name": torch.cuda.get_device_name(i), "total_memory": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    try:
        info["gpu_status"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        info["gpu_status"] = None
    return info


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def seed_everything(seed, threads=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype="uint32"), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class StatefulSampler:
    """Single-process deterministic sampler. Cursor and permutation are checkpointed."""
    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.cursor, self.epoch = 0, 0

    def take(self, n):
        out = []
        for _ in range(n):
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
                self.cursor = 0; self.epoch += 1
            out.append(self.order[self.cursor]); self.cursor += 1
        return out

    def state_dict(self):
        return {"order": self.order, "cursor": self.cursor, "epoch": self.epoch, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.order, self.cursor, self.epoch = state["order"], state["cursor"], state["epoch"]
        self.generator.set_state(state["generator"].cpu())


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[key].copy_(value)


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    # Local checkpoints contain tensors and primitive state only; no arbitrary pickle execution.
    return torch.load(path, map_location=device, weights_only=True)


def model_from_checkpoint(path, device="cpu", ema=True):
    from ..config import config_from_dict
    from ..models.network import EvidenceSR
    checkpoint = load_checkpoint(path, "cpu")
    config = config_from_dict(checkpoint["config"])
    model = EvidenceSR(config)
    model.load_state_dict(checkpoint["ema"] if ema else checkpoint["model"])
    return model.to(device).eval(), checkpoint
