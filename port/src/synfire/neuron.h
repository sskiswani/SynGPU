#ifndef PORT_NEURON_H
#define PORT_NEURON_H

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class Neuron {
  public:
    Neuron();

    Neuron( int label, double exc_freq, double inh_freq, double exc_amp, double inh_amp, double global_inhibition );


    bool Update( float dt );

    void ExciteInhibit( char p, double amp );

    void Reset();

    void RecordSpike( double t );

    void ResetSpike();

    double Get(char code) {
        if(code == 'e') return _gexc;
        else if (code == 'i') return _ginh;
        else if (code == 'v') return _volts;
    }

    double Volts() { return _volts; }
    double Excitatory() { return _gexc; }
    double Inhibitory() { return _ginh; }

  private:
    void Initialize( int label,
                     double exc_freq,
                     double inh_freq,
                     double exc_amp,
                     double inh_amp,
                     double global_inhibition,
                     double leak );

    void SetSpinFrequency( double excitory, double inhibitory );

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ INTERNAL
    int _label;                     // The Neuron's label.
    double _volts, _gexc, _ginh;    // Membrane potential, excitory and inibitory conductances
    double _global_in;              // Global inhibition amplitude.
    double _spfreq_ex, _spamp_ex;   // Spontaneous excitory frequency and amplitude.
    double _spfreq_in, _spamp_in;   // Spontaneous inhibitory frequency and amplitude.

    double *_spkhist;               // history of spike times.
    double _LEAKREV;                // TODO: not sure what this means.

    // Two factors that limit development of cycles
    //  _cRef - refactory period 25 ms
    //  _cLatent - LTD (longterm depression) 20 ms
    int _cLatent, _cRef, _cSpike;   // Latent, refractory, and spike counters.


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
