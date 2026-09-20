/*
 * Gamma random variate generation for queueing-model service-time studies.
 */
#ifndef GAMMA_RANDOM_H
#define GAMMA_RANDOM_H

double gamma_random(double shape, double scale);
double gamma_random_from_mean_cov(double mean, double cov);

#endif
