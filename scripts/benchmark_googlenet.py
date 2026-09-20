"""
Benchmark real GoogLeNet ONNX inference with ONNX Runtime CUDAExecutionProvider.

Peak memory is sampled NVIDIA device memory from NVML when available. It is
not an exact ONNX Runtime allocator high-water mark and is documented as such
in the output metadata.
"""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_PATH = Path("models/googlenet.onnx")
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 50
OUTPUT_PATH = Path("googlenet_gpu_benchmark.csv")
METADATA_PATH = Path("googlenet_gpu_benchmark_metadata.json")


def gpu_metadata():
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle).decode()
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion().decode()
        return name, memory.used / (1024**2), driver, pynvml, handle
    except (ImportError, RuntimeError, OSError):
        return "unknown", float("nan"), "unknown", None, None


def cuda_version():
    version = command_version(["nvcc", "--version"])
    if version != "unknown":
        return version
    try:
        import torch

        return str(torch.version.cuda)
    except (ImportError, AttributeError):
        return "unknown"


def command_version(command):
    try:
        return subprocess.check_output(command, text=True).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def git_commit():
    return command_version(["git", "rev-parse", "HEAD"])


def model_sha256():
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize_cuda():
    """Synchronize the default CUDA stream when a CUDA framework is available."""
    try:
        import torch

        torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        # ONNX Runtime synchronizes completed Run calls before returning.
        pass


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run scripts/download_googlenet.sh first."
        )

    gpu_name, baseline_memory, driver, pynvml, handle = gpu_metadata()
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu "
            "and verify the Kaggle CUDA runtime."
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    session.disable_fallback()
    active_providers = session.get_providers()
    if "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            f"CUDAExecutionProvider was not activated; providers={active_providers}"
        )

    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = [str(value) for value in input_info.shape]
    print(f"GPU name: {gpu_name}")
    print("CUDA available: yes")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Execution providers: {active_providers}")
    print(f"Input shape template: {input_shape}")
    print(f"Model path: {MODEL_PATH}")

    rows = []
    for batch_size in BATCH_SIZES:
        input_data = np.random.default_rng(20240921 + batch_size).random(
            (batch_size, 3, 224, 224), dtype=np.float32
        )
        for _ in range(WARMUP_RUNS):
            session.run(None, {input_name: input_data})
        synchronize_cuda()

        latencies = []
        peak_memory = baseline_memory
        for _ in range(MEASUREMENT_RUNS):
            synchronize_cuda()
            start = time.perf_counter()
            session.run(None, {input_name: input_data})
            synchronize_cuda()
            latencies.append((time.perf_counter() - start) * 1000.0)
            if pynvml is not None:
                peak_memory = max(
                    peak_memory,
                    pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2),
                )

        mean_latency = float(np.mean(latencies))
        rows.append(
            {
                "gpu_name": gpu_name,
                "batch_size": batch_size,
                "mean_latency_ms": mean_latency,
                "std_latency_ms": float(np.std(latencies, ddof=1)),
                "throughput_img_s": batch_size / (mean_latency / 1000.0),
                "peak_memory_mb": peak_memory,
            }
        )
        print(rows[-1])

    with OUTPUT_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "gpu_name": gpu_name,
        "cuda_available": True,
        "cuda_version": cuda_version(),
        "driver_version": driver,
        "compute_capability": command_version(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        ),
        "onnxruntime_version": ort.__version__,
        "python_version": platform.python_version(),
        "compiler_version": command_version(["cc", "--version"]),
        "git_commit": git_commit(),
        "input_shape": ["batch_size", 3, 224, 224],
        "input_dtype": "float32",
        "model_path": str(MODEL_PATH),
        "model_sha256": model_sha256(),
        "batch_sizes": BATCH_SIZES,
        "warmup_runs": WARMUP_RUNS,
        "measurement_runs": MEASUREMENT_RUNS,
        "input_random_seed_formula": "20240921 + batch_size",
        "memory_metric": (
            "NVML device used-memory samples; includes other processes and "
            "is not an exact ONNX Runtime allocator peak."
        ),
        "execution_providers": active_providers,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH} and {METADATA_PATH}")


if __name__ == "__main__":
    main()
