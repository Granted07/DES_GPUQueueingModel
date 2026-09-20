# DES GPU Queueing Model

Research-oriented discrete-event simulator for GPU batch inference queues with
finite capacity, `(a,b)` batching, synthetic semantic-cache hits, warm-up
removal, and Little's Law diagnostics.

## Project layout

```text
include/       Public simulator interfaces and data types
src/           Simulation core and sweep executable
tests/         Deterministic regression tests
docs/          Research notes and experiment documentation
results/       Recommended location for generated CSV output
```

The implementation uses fixed-size storage and does not allocate memory
dynamically during simulation.

## Build

```text
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the sweep executable from the project directory:

```text
build/DES_GPUQueingModel
```

It writes `queue_cache_results.csv` with one row for every cache probability,
batch limit, and independent seed combination.

## Kaggle NVIDIA GPU benchmark

The repository includes an optional CUDA executable,
`cuda_gpu_benchmark`. It is enabled automatically when CMake detects `nvcc`
and the CUDA toolkit. On a Kaggle GPU notebook, install/build with:

```text
!apt-get update -qq
!apt-get install -y -qq cmake build-essential
!cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
!cmake --build build --target cuda_gpu_benchmark -j2
!./build/cuda_gpu_benchmark
!pip install -r requirements.txt
!python plotter_distributions.py
```

The executable queries the assigned NVIDIA device using CUDA runtime APIs,
measures a repeatable CUDA kernel workload for batch sizes
`1, 2, 4, 8, 16, 32`, and writes:

- `gpu_benchmark_predictions.csv`: measured latency, derived throughput, and
  allocated workload memory for the actual GPU.
- `gpu_benchmark_cov_comparison.csv`: the existing DES metrics driven by a
  linear service model fitted to those measurements for CoV values `0.0`,
  `0.5`, and `1.4`.

These are measurements of the included proxy workload, not a claim of
application-specific inference performance. The GPU name and compute
capability are printed at runtime. If CUDA is unavailable, the existing C
simulator and placeholder/reference benchmark targets remain buildable.

## Reproducibility

Each repetition has an explicit seed. The simulator uses the C standard
library RNG and preserves the no-cache random stream when `p_hit == 0.0` or
cache simulation is disabled. Record the executable version, configuration
constants, compiler, and generated CSV together for each experiment.

## Metrics

The output includes overall latency, miss-only latency, mean queue length,
blocking probability, effective cache-hit probability, effective arrival
rate, and the Little's Law latency cross-check.
