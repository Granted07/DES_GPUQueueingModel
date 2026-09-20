# BatOpt comparison protocol

This repository compares a measured GoogLeNet GPU service curve with a
BatOpt-style finite-capacity dynamic-batching simulation. The stochastic
roles are distinct:

1. Requests arrive as a Poisson process, so inter-arrival times are
   exponentially distributed.
2. GPU inference/service time is deterministic when `CoV=0`, or Gamma
   distributed when `CoV=0.5` or `CoV=1.4`.
3. Gamma parameters use `shape=1/CoV^2` and `scale=mean*CoV^2`.

| Metric | BatOpt | Our implementation |
|---|---|---|
| GPU | Paper hardware configuration | NVIDIA GPU assigned by Kaggle |
| Model | BatOpt paper workload/model | GoogLeNet ONNX |
| Input size | Paper-defined | `[batch, 3, 224, 224]`, float32 |
| Batch sizes | Paper experiment values | 1, 2, 4, 8, 16, 32, 64 |
| Arrival process | Poisson | Poisson |
| Inter-arrival distribution | Exponential | Exponential |
| Service-time distribution | Gamma in simulation scenarios | Deterministic or Gamma |
| CoV values | 0.5 and 1.4 scenarios | 0, 0.5, and 1.4 |
| Queue capacity | Paper configuration | Configured in DES experiment |
| Repetitions | Paper methodology | Five DES repetitions per configuration |
| Latency metric | Paper-defined inference latency | DES expected wait/service latency |
| Throughput metric | Paper throughput | Measured images/sec from ONNX Runtime |
| Memory metric | Paper memory methodology | NVML sampled device used memory; not exact ORT allocator peak |
| Dynamic batching policy | BatOpt policy | Existing `(a,b)` finite-queue batch trigger |

Results must not be interpreted as outperforming BatOpt unless hardware,
model, workload, queue settings, and metrics are made comparable. In
particular, a Kaggle Tesla T4 differs from the paper's hardware and the model
input/workload may differ.

The real benchmark records GPU name, CUDA/driver information, ONNX Runtime
version, Python version, input shape, model SHA-256, batch sizes, warm-up and
measurement counts. The DES records its arrival rate, queue capacity,
service CoV values, and deterministic seeds in its source configuration.
