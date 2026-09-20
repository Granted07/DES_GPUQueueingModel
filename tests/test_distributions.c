/*
 * Statistical regression tests for the production exponential and Gamma
 * random variate generators.
 */
#include "gamma_random.h"
#include "sim_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_COUNT 100000
#define RELATIVE_TOLERANCE 0.02
#define SHAPE_LT_ONE_TOLERANCE 0.05

static int close_relative(double actual, double expected, double tolerance)
{
    return fabs(actual - expected) / expected <= tolerance;
}

static void sample_moments(double (*sampler)(void *context),
                           void *context,
                           double *mean,
                           double *variance)
{
    double sum = 0.0;
    double sum_squared = 0.0;
    int index;
    for (index = 0; index < SAMPLE_COUNT; index++) {
        double value = sampler(context);
        sum += value;
        sum_squared += value * value;
    }
    *mean = sum / (double)SAMPLE_COUNT;
    *variance = sum_squared / (double)SAMPLE_COUNT - *mean * *mean;
}

static double exponential_sample(void *context)
{
    return exponential_interarrival(*(double *)context);
}

typedef struct {
    double mean;
    double cov;
} GammaContext;

static double gamma_sample(void *context)
{
    GammaContext *gamma = (GammaContext *)context;
    return gamma_random_from_mean_cov(gamma->mean, gamma->cov);
}

static int test_exponential(void)
{
    double lambda = 0.05;
    double mean;
    double variance;
    double sample_cov;
    int passed;

    srand(20240921U);
    sample_moments(exponential_sample, &lambda, &mean, &variance);
    sample_cov = sqrt(variance) / mean;
    passed = close_relative(mean, 1.0 / lambda, RELATIVE_TOLERANCE) &&
             close_relative(variance, 1.0 / (lambda * lambda),
                            RELATIVE_TOLERANCE) &&
             close_relative(sample_cov, 1.0, RELATIVE_TOLERANCE);
    printf("%s exponential: mean=%.6f expected=%.6f variance=%.6f "
           "expected=%.6f cov=%.6f expected=1.000000\n",
           passed ? "PASS" : "FAIL", mean, 1.0 / lambda, variance,
           1.0 / (lambda * lambda), sample_cov);
    return passed;
}

static int test_gamma_case(double requested_mean, double requested_cov,
                           unsigned int seed)
{
    GammaContext context = {requested_mean, requested_cov};
    double mean;
    double variance;
    double sample_cov;
    double expected_variance = (requested_mean * requested_cov) *
                               (requested_mean * requested_cov);
    double tolerance = requested_cov > 1.0
        ? SHAPE_LT_ONE_TOLERANCE
        : RELATIVE_TOLERANCE;
    int passed;

    srand(seed);
    sample_moments(gamma_sample, &context, &mean, &variance);
    sample_cov = sqrt(variance) / mean;
    passed = close_relative(mean, requested_mean, tolerance) &&
             close_relative(variance, expected_variance, tolerance) &&
             close_relative(sample_cov, requested_cov, tolerance);
    if (requested_cov == 1.0) {
        passed = passed &&
                 close_relative(variance, requested_mean * requested_mean,
                                tolerance);
    }
    printf("%s gamma(mean=%.1f,cov=%.1f): mean=%.6f expected=%.6f "
           "variance=%.6f expected=%.6f cov=%.6f expected=%.6f\n",
           passed ? "PASS" : "FAIL", requested_mean, requested_cov, mean,
           requested_mean, variance, expected_variance, sample_cov,
           requested_cov);
    return passed;
}

int main(void)
{
    int passed = test_exponential();
    passed = test_gamma_case(20.0, 0.5, 20240922U) && passed;
    passed = test_gamma_case(20.0, 1.0, 20240923U) && passed;
    passed = test_gamma_case(20.0, 1.4, 20240924U) && passed;
    return passed ? 0 : 1;
}
