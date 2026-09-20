/*
 * Actual NVIDIA GPU benchmark for Kaggle CUDA notebooks.
 *
 * This measures a repeatable CUDA kernel workload, reports the assigned GPU,
 * fits the measured batch latency to the DES linear service model, and runs
 * the existing queue simulator with those measured service parameters.
 */
#include "sim_core.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWarmupIterations = 10;
constexpr int kTimedIterations = 50;
constexpr int kWorkPerImage = 262144;
constexpr int kMaxBatchSize = 32;
constexpr int kDesRepetitions = 5;
constexpr int kDesArrivals = 10000;
constexpr int kDesWarmup = 1000;

struct MeasuredPoint {
    int batch_size;
    double latency_ms;
    double memory_mb;
};

__global__ void inference_proxy_kernel(float *data, int elements,
                                       int iterations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    float value = data[index];
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = value * 1.000001f + 0.000001f;
    }
    data[index] = value;
}

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

double measure_batch_latency(int batch_size, double *memory_mb)
{
    const int elements = batch_size * kWorkPerImage;
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    float *device_data = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;

    check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_data), bytes),
               "cudaMalloc");
    check_cuda(cudaMemset(device_data, 0, bytes), "cudaMemset");
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        inference_proxy_kernel<<<blocks, threads>>>(device_data, elements, 100);
    }
    check_cuda(cudaGetLastError(), "warmup kernel launch");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < kTimedIterations; ++iteration) {
        inference_proxy_kernel<<<blocks, threads>>>(device_data, elements, 100);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime");

    *memory_mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cuda(cudaFree(device_data), "cudaFree");
    return static_cast<double>(elapsed_ms) / kTimedIterations;
}

void fit_linear_service_model(const std::vector<MeasuredPoint> &points,
                              double *slope, double *intercept)
{
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    const double count = static_cast<double>(points.size());

    for (const MeasuredPoint &point : points) {
        sum_x += point.batch_size;
        sum_y += point.latency_ms;
        sum_xx += point.batch_size * point.batch_size;
        sum_xy += point.batch_size * point.latency_ms;
    }
    const double denominator = count * sum_xx - sum_x * sum_x;
    if (denominator == 0.0) {
        *slope = 0.0;
        *intercept = points.empty() ? 0.0 : points.front().latency_ms;
        return;
    }
    *slope = (count * sum_xy - sum_x * sum_y) / denominator;
    *intercept = (sum_y - *slope * sum_x) / count;
}

int write_queueing_results(const char *gpu_name, double service_slope,
                           double service_intercept, int max_batch_size)
{
    const double cov_values[] = {0.0, 0.5, 1.4};
    FILE *output = std::fopen("gpu_benchmark_cov_comparison.csv", "w");
    if (output == nullptr) {
        std::perror("gpu_benchmark_cov_comparison.csv");
        return EXIT_FAILURE;
    }
    std::fprintf(output, "gpu_name,batch_size,cov,mean_E_W,mean_E_L,"
                         "mean_P_block\n");

    for (int batch_size = 1; batch_size <= max_batch_size; batch_size *= 2) {
        for (double cov : cov_values) {
            double sum_wait = 0.0;
            double sum_queue = 0.0;
            double sum_block = 0.0;
            for (int repetition = 0; repetition < kDesRepetitions;
                 ++repetition) {
                SimulationConfig config = {};
                config.lambda = 0.02;
                config.a = 4;
                config.b = batch_size;
                config.service_slope = service_slope;
                config.service_intercept = service_intercept;
                config.queue_capacity = MAX_QUEUE_CAPACITY;
                config.arrivals_to_generate = kDesArrivals + kDesWarmup;
                config.seed = static_cast<unsigned int>(
                    900000 + batch_size * 100 + static_cast<int>(cov * 10) +
                    repetition);
                config.p_hit = 0.0;
                config.cache_lookup_latency = 0.5;
                config.enable_cache_sim = 0;
                config.warmup_arrivals = kDesWarmup;
                config.service_time_cov = cov;
                SimulationResults results = simulate(&config);
                sum_wait += results.expected_wait_miss_only;
                sum_queue += results.expected_number_in_queue;
                sum_block += results.blocking_probability;
            }
            std::fprintf(output, "%s,%d,%.1f,%.10f,%.10f,%.10f\n",
                         gpu_name, batch_size, cov,
                         sum_wait / kDesRepetitions,
                         sum_queue / kDesRepetitions,
                         sum_block / kDesRepetitions);
        }
    }
    std::fclose(output);
    return EXIT_SUCCESS;
}

}  // namespace

int main()
{
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA-capable NVIDIA GPU was detected.\n");
        return EXIT_FAILURE;
    }

    int device = 0;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties = {};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");

    std::printf("Actual CUDA GPU: %s (compute capability %d.%d, %.1f GiB)\n",
                properties.name, properties.major, properties.minor,
                static_cast<double>(properties.totalGlobalMem) /
                    (1024.0 * 1024.0 * 1024.0));

    std::vector<MeasuredPoint> points;
    for (int batch_size = 1; batch_size <= kMaxBatchSize; batch_size *= 2) {
        double memory_mb = 0.0;
        double latency_ms = measure_batch_latency(batch_size, &memory_mb);
        points.push_back({batch_size, latency_ms, memory_mb});
        std::printf("batch=%d latency_ms=%.6f allocated_memory_mb=%.6f\n",
                    batch_size, latency_ms, memory_mb);
    }

    double service_slope = 0.0;
    double service_intercept = 0.0;
    fit_linear_service_model(points, &service_slope, &service_intercept);
    std::printf("DES service model: slope=%.6f ms/task intercept=%.6f ms\n",
                service_slope, service_intercept);

    FILE *prediction_output = std::fopen("gpu_benchmark_predictions.csv", "w");
    if (prediction_output == nullptr) {
        std::perror("gpu_benchmark_predictions.csv");
        return EXIT_FAILURE;
    }
    std::fprintf(prediction_output,
                 "gpu_name,batch_size,predicted_latency_ms,"
                 "predicted_throughput_img_s,predicted_memory_mb\n");
    for (const MeasuredPoint &point : points) {
        const double throughput = point.batch_size /
                                  (point.latency_ms / 1000.0);
        std::fprintf(prediction_output, "%s,%d,%.10f,%.10f,%.10f\n",
                     properties.name, point.batch_size, point.latency_ms,
                     throughput, point.memory_mb);
    }
    std::fclose(prediction_output);

    return write_queueing_results(properties.name, service_slope,
                                  service_intercept, kMaxBatchSize);
}
