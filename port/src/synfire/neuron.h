#ifndef PORT_NEURON_H
#define PORT_NEURON_H

#ifdef __GPU_BUILD__
#include "cuda_utils.h"
#include <cuda_runtime.h>
#define CUDA_CALLABLE __host__ __device__
#define INLINE
#else
#define CUDA_CALLABLE
#define INLINE  inline
#endif

class Neuron {
  public:
#ifdef __GPU_BUILD__
    friend class CUSynfire;
#endif

    Neuron();

    Neuron( int label, double exc_freq, double inh_freq, double exc_amp, double inh_amp, double global_inhibition );


    CUDA_CALLABLE bool Update( float dt );

    CUDA_CALLABLE void neur_dyn( double dt, bool no_volt );

    CUDA_CALLABLE void Reset();

    CUDA_CALLABLE void ExciteInhibit( double amp, char p );

    CUDA_CALLABLE INLINE double Get( char code ) {
        if (code == 'e') return _gexc;
        else if (code == 'i') return _ginh;
        else if (code == 'v') return _volts;

        // Error.
        return 0.0;
    }

    CUDA_CALLABLE INLINE double Volts() { return _volts; }

    CUDA_CALLABLE INLINE double Excitatory() { return _gexc; }

    CUDA_CALLABLE INLINE double Inhibitory() { return _ginh; }

  private:
    CUDA_CALLABLE void Initialize( int label,
                                   double exc_freq,
                                   double inh_freq,
                                   double exc_amp,
                                   double inh_amp,
                                   double global_inhibition,
                                   double leak );

    CUDA_CALLABLE INLINE void SetSpinFrequency( double excitory, double inhibitory ) {
        _spfreq_ex = excitory * 0.001;
        _spfreq_in = inhibitory * 0.001;
    }

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ INTERNAL
    int _label;                     // The Neuron's label.
    double _volts, _gexc, _ginh;    // Membrane potential, excitory and inibitory conductances
    double _global_in;              // Global inhibition amplitude.
    double _spfreq_ex, _spamp_ex;   // Spontaneous excitory frequency and amplitude.
    double _spfreq_in, _spamp_in;   // Spontaneous inhibitory frequency and amplitude.

    double _LEAKREV;                // TODO: not sure what this means.

    // Two factors that limit development of cycles
    //  _cRef - refactory period 25 ms
    //  _cLatent - LTD (longterm depression) 20 ms
    int _cLatent, _cRef;   // Latent and refractory counters.


    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ CONSTANTS
    // Integrate and Fire model parameters. (in ms)
    const static double DECAY_INHIBITORY;        // Inhibitory Decay
    const static double DECAY_EXCITORY;          // Excitory Decay
    const static double DECAY_MEMBRANE;          // Membrane Decay
    const static double SPIKE_THRESHOLD;         // Spike threshold (mV)
    const static double RESET;                   // Membrane reset potential (mV)
    const static double INHIBITORY_REVERSAL;     // potential (mV)
    const static double REFCTORY_TIME;           // ms
    const static double LATENCY_TIME;            // ms

    // Spontaneous activity defaults.
    const static double
            DEFAULT_EXFREQ,
            DEFAULT_INFREQ,
            DEFAULT_EXAMP,
            DEFAULT_INAMP,
            DEFAULT_GLOBAL_I,
            DEFAULT_INHDECAY,
            DEFAULT_LEAK;

    static int _LABEL_COUNTER;
};

#endif //PORT_NEURON_H
