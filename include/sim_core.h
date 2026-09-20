#ifndef SIM_CORE_H
#define SIM_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "sim_types.h"

SimulationResults simulate(const SimulationConfig *config);
double exponential_interarrival(double lambda);

#ifdef __cplusplus
}
#endif

#endif
