#ifndef PORT_SYNAPSE_H
#define PORT_SYNAPSE_H

#include "helpers.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class Neuron;

class Synapses {
  public:
    Synapses( double fract_act,
              double glob,
              double act_thres,
              double sup_thres,
              double syn_max,
              double syn_decay,
              int form_opt,
              int network_size,
              int tempNSS,
              double window,
              double eq_syn );

    CUDA_CALLABLE void Activate( char p, int pre, int post );

    CUDA_CALLABLE void Deactivate( char p, int pre, int post );

    CUDA_CALLABLE void Synaptic_Plasticity( int spiker, double t, double *spk_times, int spk_count );

    CUDA_CALLABLE double PotentiationFunc( double time, int spk_count, double *hist, char pd_type );

    CUDA_CALLABLE void CheckThreshold( double syn_str, int pre, int post, char syn_type, char pd_type );

    CUDA_CALLABLE void SynapticDecay();

    CUDA_CALLABLE double GetPostSynapticLabel( char syn_type, int pre, bool *&post_arr );

    // Accessors
    CUDA_CALLABLE int CountSynapses( char syn_type );

    CUDA_CALLABLE inline double GetNSS( int label ) { return _NSS[label]; }

//    inline double GetSynapticStrength( int pre, int post ) { return _G[pre * _size + post]; }

    /**
     * @param pre   Label of synapse start neuron.
     * @param post  Label of synapse end neuron.
     */
    CUDA_CALLABLE inline double GetSynapticStrength( int pre, int post ) { return _G(post, pre); }

    inline double *GetSynapseStartingAt( int pre ) { return _G.row(pre); }

    inline int GetActCount( int i ) { return _actcount[i]; }

    inline int GetSupCount( int i ) { return _supcount[i]; }

  private:
    // Synaptic Plasticity parameters
    const static double AMPLITUDE_LTP;      // Long-term Plasticity amplitude
    const static double AMPLITUDE_LTD;      // Long-term Depression amplitude
    const static double INV_DECAY_LTP;      // Long-term Plasticity Inverse Decay times (ms)
    const static double INV_DECAY_LTD;      // Long-term Depression Inverse Decay times (ms)
    const static double POTENTIATION_FUNT;  // Potentiation
    const static double DEPRESSION_FUNT;    // Depression

    //~ Internal data.
    int _size;  // size of the network.
    double _window; // history time window size (ms)
    double _actthres, _supthres, _synmax;   // Active, super thresholds, synapse cap
    double _syndec, _GLTP;                  //synaptic decay

    int *_actcount, *_supcount, *_NSS;  // Arrays tracking numbers of active and super cns of each neuron in the network
    TArray2<double> _G;                // Synaptic strength matrix;
    TArray2<bool> _actsyn, _supsyn;     // Arrays containing the postsynaptic cns of each neuron in the network

};

#endif //PORT_SYNAPSE_H
