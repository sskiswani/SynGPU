#ifndef PORT_SYNAPSE_H
#define PORT_SYNAPSE_H

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class Synapses {

private:
    // Synaptic Plasticity parameters
    const static double AMPLITUDE_LTP;			// Long-term Plasticity amplitude
    const static double AMPLITUDE_LTD;			// Long-term Depression amplitude
    const static double INV_DECAY_LTP;			// Long-term Plasticity Inverse Decay times(ms)
    const static double INV_DECAY_LTD;			// Long-term Depression Inverse Decay times(ms)
    const static double POTENTIATION_FUNT;		// Potentiation
    const static double DEPRESSION_FUNT;		// Depression
};

#endif //PORT_SYNAPSE_H
