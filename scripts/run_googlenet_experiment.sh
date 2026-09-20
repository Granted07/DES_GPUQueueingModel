#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p models
if [[ ! -f models/googlenet.onnx ]]; then
  scripts/download_googlenet.sh
fi

python3 -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
if [[ ! -x build/cuda_gpu_benchmark ]]; then
  echo "CUDA benchmark target was not built; CUDA toolkit/nvcc is required." >&2
  exit 1
fi

python3 scripts/benchmark_googlenet.py
python3 scripts/run_des_googlenet.py
python3 analysis/compare_arrival_service_distributions.py
python3 plotter_distributions.py
