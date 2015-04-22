#include "synapses.h"

// Synaptic Plasticity parameters
const double Synapses::AMPLITUDE_LTP = .01;			// Long-term Plasticity amplitude
const double Synapses::AMPLITUDE_LTD = .0105;		// Long-term Depression amplitude

const double Synapses::INV_DECAY_LTP = .05;			// Long-term Plasticity Inverse Decay times(ms)
const double Synapses::INV_DECAY_LTD = .05;			// Long-term Depression Inverse Decay times(ms)

const double Synapses::POTENTIATION_FUNT = 5.0;		// Potentiation
const double Synapses::DEPRESSION_FUNT = 5.25;		// Depression
