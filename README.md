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
`1, 2, 4, 8, 16, 32, 64`, and writes:

- `gpu_benchmark_predictions.csv`: measured latency, derived throughput, and
  allocated workload memory for the actual GPU.
- `gpu_benchmark_cov_comparison.csv`: the existing DES metrics driven by a
  linear service model fitted to those measurements for CoV values `0.0`,
  `0.5`, and `1.4`.

These are measurements of the included proxy workload, not a claim of
application-specific inference performance. The GPU name and compute
capability are printed at runtime. If CUDA is unavailable, the existing C
simulator and placeholder/reference benchmark targets remain buildable.

## Reproducible GoogLeNet/BatOpt-style experiment

For a Kaggle GPU notebook, run:

```text
!git clone <repository-url>
%cd DES_GPUQueingModel
!chmod +x scripts/*.sh
!scripts/run_googlenet_experiment.sh
```

The script downloads `models/googlenet.onnx`, installs Python dependencies,
builds and tests the C simulator, runs the real ONNX Runtime CUDA benchmark,
feeds measured service means into the DES, generates distribution analysis,
and runs the plotting script.

The real benchmark requires `onnxruntime-gpu` and verifies that
`CUDAExecutionProvider` is active. It measures GoogLeNet inference for batch
sizes 1, 2, 4, 8, 16, 32, and 64. It records metadata in
`googlenet_gpu_benchmark_metadata.json`.

Generated CSV files include:

- `googlenet_gpu_benchmark.csv`
- `googlenet_des_cov_results.csv`
- `gpu_benchmark_cov_comparison.csv`
- `gpu_benchmark_predictions.csv`
- `queue_cache_results.csv`

Generated PNG files include:

- `googlenet_gpu_benchmark.png`
- `dist_exponential_interarrival.png`
- `dist_gamma_service_cov.png`
- `dist_exponential_vs_gamma.png`
- `googlenet_cov_latency.png`
- `googlenet_cov_blocking.png`
- `googlenet_cov_comparison.png`
- the existing distribution and CoV plot outputs

The benchmark metadata records the GPU, CUDA and ONNX Runtime versions,
Python version, model hash, batch sizes, warm-up count, measurement count,
and memory-measurement caveat. Results from a Kaggle Tesla T4 must not be
presented as direct hardware-equivalent results to BatOpt without noting the
hardware and workload differences.

## Reproducibility

Each repetition has an explicit seed. The simulator uses the C standard
library RNG and preserves the no-cache random stream when `p_hit == 0.0` or
cache simulation is disabled. Record the executable version, configuration
constants, compiler, and generated CSV together for each experiment.

## Metrics

The output includes overall latency, miss-only latency, mean queue length,
blocking probability, effective cache-hit probability, effective arrival
rate, and the Little's Law latency cross-check.
