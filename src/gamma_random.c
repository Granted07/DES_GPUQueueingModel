/*
 * Marsaglia-Tsang Gamma sampler and its mean/CoV parameterization.
 */
#include "gamma_random.h"

#include <math.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

static double uniform_random(void)
{
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
}

static double standard_normal(void)
{
    double u1 = uniform_random();
    double u2 = uniform_random();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

static double gamma_random_shape_at_least_one(double shape)
{
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);

    for (;;) {
        double x = standard_normal();
        double v = 1.0 + c * x;
        double u;
        if (v <= 0.0) {
            continue;
        }
        v = v * v * v;
        u = uniform_random();
        if (u < 1.0 - 0.0331 * x * x * x * x ||
            log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            return d * v;
        }
    }
}

double gamma_random(double shape, double scale)
{
    if (shape <= 0.0 || scale < 0.0) {
        return NAN;
    }
    if (scale == 0.0) {
        return 0.0;
    }
    if (shape < 1.0) {
        /*
         * Standard shape-boosting transformation for the shape < 1 branch.
         */
        double boosted = gamma_random_shape_at_least_one(shape + 1.0);
        return scale * boosted * pow(uniform_random(), 1.0 / shape);
    }
    return scale * gamma_random_shape_at_least_one(shape);
}

double gamma_random_from_mean_cov(double mean, double cov)
{
    double shape;
    double scale;

    if (cov <= 0.0) {
        return mean;
    }
    shape = 1.0 / (cov * cov);
    scale = mean * cov * cov;
    return gamma_random(shape, scale);
}
