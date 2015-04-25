#ifndef PORT_CUSYNFIRE_H
#define PORT_CUSYNFIRE_H

#include <vector>
#include <fstream>
#include "synfire_helpers.h"
#include "synapses.h"

typedef std::vector<double> row_t;
typedef std::vector<row_t> matrix_t;

class Neuron;

//~ Kernels
__global__ void SynapticDecayKernel( Synapses *dconnectivity, int syn_size );

__global__ void MembranePotentialKernel( float dt, Neuron *net, int net_size, bool *whospiked, float *dranCache );

class CUSynfire {
  public:
    static CUSynfire CreateCUSynfire();

    static CUSynfire CreateCUSynfire( int nsize );

    static CUSynfire CreateCUSynfire( int nsize, double dt, int num_trials, double trial_time );

    //~ CTOR
    CUSynfire( SynfireParameters );

    //~ DTOR
    ~CUSynfire();

    void Run();

    /**
     * Runs a single SynfireGrowth trial.
     *
     * @param tT   idk?
     * @param tTS  idk?
     * @param tMPL The timings for Membrane Potential Layer, e.g. [start, end, delta]. (?)
     * @param tSL  The timings for the Spike loop, e.g. [start, end, delta]. (?)
     * @param tTSa [description]
     *
     * @return The total elapsed time of the trial.
     */
    double RunTrial( double *tT, double *tTS, double *tMPL, double *tSpkLp, double *tTSa );

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Helpers & Accessors
    double GetAverageVoltage();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Member constants.
    const double DT, INV_DT;
    const int trials, trial_duration, trial_steps;

    bool stats_on;

    //~ Inhibition delay data.
    int dsteps;
    int *inh;

    // TODO: Not yet understood
    int group_s;

  private:
    void Initialize();

    void DoSpikeLoop();

    void DoSynapticDecay();

    void MembranePotentialLauncher();

    Synapses *CreateDeviceSynapses( Synapses *syn );

    SynfireParameters _params;

    //~ Neuron Data
    int network_size;
    Neuron *_network, *_dnetwork;

    //~ Neuron helpers
    float *_ranCache, *_dranCache;
    bool *_spikeFlags, *_dspikeFlags;
    std::vector<int> _whospiked;  // labels of Neurons who spiked during a timestep.
    matrix_t _spikeHistory;

    //~ Synapse Data
    Synapses _connectivity, *_dconnectivity;
    Synapses _inhibition_strength, *_dinh_str;

    //~ Timestep info
    double _elapsedTime;  // total time elapsed in a trial.

    //~ Trial data.
    int _spikeCounter;

    //~ Training data
    int *_train_lab; // training labels.
    double *_train_times;
    double _train_freq; // training spike frequency (ms)^(-1)
    double _train_amp;  // training spike strength
    double _train_dur;  // training duration in ms
};

#endif //PORT_CUSYNFIRE_H

