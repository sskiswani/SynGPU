#ifndef PORT_NEURON_H
#define PORT_NEURON_H

#include "synfire_constants.h"

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

    ~Neuron();

    bool Update( float dt );

    CUDA_CALLABLE bool Update( float dt, float r1, float r2, float r3, float r4 );

    CUDA_CALLABLE void neur_dyn( double dt, bool no_volt );

    void Reset();

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
    void Initialize( int label,
                     double exc_freq,
                     double inh_freq,
                     double exc_amp,
                     double inh_amp,
                     double global_inhibition,
                     double leak );

    INLINE void SetSpinFrequency( double excitory, double inhibitory ) {
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

    static int _LABEL_COUNTER;
};

#endif //PORT_NEURON_H
