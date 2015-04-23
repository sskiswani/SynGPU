#ifndef PORT_SYNAPSE_H
#define PORT_SYNAPSE_H

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class Neuron;

class Synapses {
  public:
	friend class CUSynfire;
	friend class Synfire;
	
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

    void Activate( char p, int pre, int post );

    void Deactivate( char p, int pre, int post );

    void Synaptic_Plasticity( int spiker, double t, Neuron **net );

    double PotentiationFunc( double time, int spsc, double *hist, char pd_type );

    void CheckThreshold( double syn_str, int pre, int post, char syn_type, char pd_type );

    void SynapticDecay();

    double GetPostSynapticLabel( char syn_type, int pre, bool *&post );

    // Accessors
    inline double GetNSS( int label ) { return _NSS[label]; }

    inline double GetSynapticStrength( int pre, int post ) { return _G[pre * _size + post]; }

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
    double *_G; // Synaptic strength matrix;
    int *_actcount, *_supcount, *_NSS;      // Arrays tracking numbers of active and super cns of each neuron in the network
    double _actthres, _supthres, _synmax;   // Active, super thresholds, synapse cap
    double _syndec, _GLTP;                  //synaptic decay

    bool *_actsyn, *_supsyn;             // Arrays containing the postsynaptic cns of each neuron in the network
};

#endif //PORT_SYNAPSE_H
