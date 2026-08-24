"""Reproducible experiment I/O shared by IST v0.2 levels."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(30):
        try:
            os.replace(temporary, path); return
        except PermissionError:
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    for attempt in range(20):
        try:
            os.replace(temporary, path); return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    torch.save(payload, path)
    temporary.unlink(missing_ok=True)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(device: torch.device, seed=None) -> dict:
    return {"seed": seed, "git_commit": git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "device": str(device), "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform(),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tensor_to_json(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_json(item) for item in value]
    return value
