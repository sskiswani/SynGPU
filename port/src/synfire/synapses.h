#ifndef PORT_SYNAPSE_H
#define PORT_SYNAPSE_H

#include "utility.h"
#include "helpers.h"

#ifdef __GPU_BUILD__
#include "cuda_utils.h"
#include <cuda_runtime.h>

#define INC(x)  (atomicAdd(&x, 1))
#define DEC(x)  (atomicSub(&x, 1))

#define CUDA_CALLABLE __host__ __device__
#define INLINE
#else
#define INC(x)  (++(x))
#define DEC(x)  (--(x))
#define CUDA_CALLABLE
#define INLINE  inline
#endif

class Neuron;

class Synapses {
  public:
#ifdef __GPU_BUILD__
    friend class CUSynfire;
    friend void SynapticDecayKernel( Synapses *dconnectivity, int syn_size );
#endif

    Synapses() {
    }

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

    ~Synapses();

    /**
     *  Activate a synapse between two neurons.
     *
     *  @param p        'a' for active, 's' for super
     *  @param pre      pre-neuron label
     *  @param post     post-neuron label
     */
    CUDA_CALLABLE void Activate( char p, int pre, int post );

    /**
     *  Deactivate a synapse between two neurons.
     *
     *  @param p        'a' for active, 's' for super
     *  @param pre      pre-neuron label
     *  @param post     post-neuron label
     */
    CUDA_CALLABLE void Deactivate( char p, int pre, int post );

    /**
     * Synaptic_Plasticity.
     *
     * @param spiker        Label of the neuron that spiked.
     * @param t             Elapsed time of the current trial.
     * @param spk_times     A log of the times the spiking neuron has previous spiked.
     * @param spk_count     Size of the spk_times array.
     */
    CUDA_CALLABLE void Synaptic_Plasticity( int spiker, double t, double *spk_times, int spk_count );

    /**
     * PotentiationFunc
     *
     * @param time          Elapsed time of the trial.
     * @param spk_count     Size of the spk_times array.
     * @param spk_times     A log of the times the spiking neuron has previous spiked.
     * @param pd_type       'p' for potentiation, 'd' for depression
     */
    CUDA_CALLABLE double PotentiationFunc( double time, int spk_count, double *hist, char pd_type );

    /**
     * Ensure that the synaptic strength of (pre, post) is within threshold limit.
     *
     * @param syn_str   The new synapse strength.
     * @param pre       Label of synapse start neuron.
     * @param post      Label of synapse end neuron.
     * @param syn_type  'a' for active, 's' for super
     * @param pd_type   'p' for potentiation, 'd' for depression
     */
    CUDA_CALLABLE void CheckThreshold( double syn_str, int pre, int post, char syn_type, char pd_type );

    void SynapticDecay();

    CUDA_CALLABLE double GetPostSynapticLabel( char syn_type, int pre, bool *&post_arr );

    // Accessors
    int CountSynapses( char syn_type );

    CUDA_CALLABLE INLINE double GetNSS( int label ) { return _NSS[label]; }

//    inline double GetSynapticStrength( int pre, int post ) { return _G[pre * _size + post]; }

    /**
     * @param pre   Label of synapse start neuron.
     * @param post  Label of synapse end neuron.
     */
    CUDA_CALLABLE INLINE double GetSynapticStrength( int pre, int post ) { return _G(post, pre); }

    /**
     * @param pre   Label of synapse start neuron.
     * @param post  Label of synapse end neuron.
     * @param value New synaptic strength value.
     */
    CUDA_CALLABLE INLINE void SetSynapticStrength( int pre, int post, double value ) {
        _G(post, pre) = value;
    }

    CUDA_CALLABLE INLINE double *GetSynapsesStartingAt( int pre ) { return _G.row(pre); }

    CUDA_CALLABLE INLINE int GetActCount( int i ) { return _actcount[i]; }

    CUDA_CALLABLE INLINE int GetSupCount( int i ) { return _supcount[i]; }

    CUDA_CALLABLE INLINE double GetSynDecay() { return _syndec; }

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
    TArray2<bool> _actsyn, _supsyn;    // Arrays containing the postsynaptic cns of each neuron in the network

};

#endif //PORT_SYNAPSE_H
