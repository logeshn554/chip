"""
Compute Device & GPU Acceleration Detection Utility.

Ensures that if ANY GPU is present (NVIDIA CUDA, AMD ROCm/HIP, Intel Iris/Arc XPU,
Apple Metal MPS, or Windows DirectML), it is automatically discovered and leveraged
for both local Ollama LLM inference offloading and PyTorch/Transformers execution.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def is_gpu_disabled() -> bool:
    """Check if the user explicitly disabled GPU via environment variables."""
    val = os.environ.get("USE_GPU", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return True
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_vis is not None and cuda_vis.strip() in ("", "-1"):
        return True
    return False


def detect_system_gpus() -> list[dict[str, Any]]:
    """Detect all video controllers / GPUs installed on the host OS."""
    gpus: list[dict[str, Any]] = []

    # 1. Check PyTorch backends if torch is importable
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory
                gpus.append({
                    "name": name,
                    "backend": "cuda",
                    "vram_bytes": mem,
                    "vram_mb": round(mem / (1024 * 1024), 2),
                    "device_index": i,
                })

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            gpus.append({
                "name": "Apple Silicon MPS",
                "backend": "mps",
                "vram_bytes": 0,
                "vram_mb": 0,
            })

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                gpus.append({
                    "name": torch.xpu.get_device_name(i),
                    "backend": "xpu",
                    "vram_bytes": 0,
                    "vram_mb": 0,
                    "device_index": i,
                })
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"PyTorch GPU check returned: {e}")

    if gpus:
        return gpus

    # 2. Check Windows WMI / CimInstance
    if platform.system() == "Windows":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", 
                   "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if proc.returncode == 0 and proc.stdout.strip():
                import json
                raw = json.loads(proc.stdout)
                controllers = [raw] if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
                for ctrl in controllers:
                    name = ctrl.get("Name") or "Unknown GPU"
                    ram = int(ctrl.get("AdapterRAM") or 0)
                    gpus.append({
                        "name": name,
                        "backend": "directml/vulkan/wmi",
                        "vram_bytes": ram,
                        "vram_mb": round(ram / (1024 * 1024), 2),
                    })
        except Exception as e:
            logger.debug(f"Windows CimInstance check error: {e}")

    # 3. Check Linux nvidia-smi / lspci
    elif platform.system() == "Linux":
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if proc.returncode == 0 and proc.stdout.strip():
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    gpus.append({
                        "name": parts[0],
                        "backend": "nvidia-smi",
                        "vram_str": parts[1] if len(parts) > 1 else "Unknown",
                    })
        except Exception:
            pass

    return gpus


def has_gpu() -> bool:
    """Return True if any hardware GPU is detected and not disabled."""
    if is_gpu_disabled():
        return False
    gpus = detect_system_gpus()
    return len(gpus) > 0


def get_ollama_gpu_options() -> dict[str, Any]:
    """Get GPU options for Ollama API calls.
    
    If GPU is available and not disabled:
    - num_gpu: -1 offloads all layers to GPU in Ollama.
    """
    if is_gpu_disabled():
        return {"num_gpu": 0}

    # If env var explicitly sets num_gpu, honor it
    if "OLLAMA_NUM_GPU" in os.environ:
        try:
            return {"num_gpu": int(os.environ["OLLAMA_NUM_GPU"])}
        except ValueError:
            pass

    # If GPU is detected on host, configure maximum layer offloading (-1)
    if has_gpu():
        return {"num_gpu": -1}

    # Default fallback to auto
    return {}


def get_torch_device_and_dtype() -> tuple[str, Any]:
    """Return optimal device string ('cuda', 'mps', 'xpu', 'cpu') and dtype for PyTorch."""
    if is_gpu_disabled():
        import torch
        return "cpu", torch.float32

    try:
        import torch

        if torch.cuda.is_available():
            # Check for bfloat16 capability (Ampere+)
            if torch.cuda.is_bf16_supported():
                return "cuda", torch.bfloat16
            return "cuda", torch.float16

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", torch.float16

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu", torch.bfloat16

        try:
            import torch_directml
            return torch_directml.device(), torch.float32
        except ImportError:
            pass

        return "cpu", torch.float32
    except ImportError:
        return "cpu", None
