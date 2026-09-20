#ifndef SIM_CORE_H
#define SIM_CORE_H

#include "sim_types.h"

SimulationResults simulate(const SimulationConfig *config);
double exponential_interarrival(double lambda);

#endif
