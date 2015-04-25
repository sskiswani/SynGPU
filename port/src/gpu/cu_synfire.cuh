#ifndef PORT_CUSYNFIRE_H
#define PORT_CUSYNFIRE_H

#include <vector>
#include <fstream>
#include "synapses.h"
#include "synfire_helpers.h"

typedef std::vector<double> row_t;
typedef std::vector<row_t> matrix_t;

class Neuron;

//"""""""""""""""""""""""""""""""""""""""""""""""""
// Kernels
//"""""""""""""""""""""""""""""""""""""""""""""""""

/**
 * Perform Synaptic Decay on the GPU.
 *
 * @param dconnectivity   The connectivity synapses object.
 * @param syn_size        Maximum size of the network.
 */
__global__ void SynapticDecayKernel( Synapses *dconnectivity, int syn_size );

/**
 * Run the Membrane Potential Layer update step on the GPU.
 *
 * @param dt          Update timestep.
 * @param net         The entire neural network.
 * @param net_size    Size of the neural net.
 * @param whospiked   Index corresponds to whether or not the neuron spiked during update.
 * @param dranCache   Preallocated array of random values for Neuron.Update() calls.
 */
__global__ void MembranePotentialKernel( float dt, Neuron *net, int net_size, bool *whospiked, float *dranCache );

class CUSynfire {
  public:
    static CUSynfire CreateCUSynfire();

    static CUSynfire CreateCUSynfire( int nsize );

    static CUSynfire CreateCUSynfire( int nsize, double dt, int num_trials, double trial_time );

    // CTOR
    CUSynfire( SynfireParameters );

    // DTOR
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

    /**
     * @return average voltage of neurons in the net.
     */
    double GetAverageVoltage();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Member constants & trial information.
    const double DT, INV_DT;
    const int trials, trial_duration, trial_steps;
    bool stats_on;

    // Inhibition delay data.
    int dsteps;
    int *inh;
    int group_s; // TODO: Not yet understood

    // TODO: Threading prefs.
//    int threadsPerBlock;

  private:
    void Initialize();

    void DoSpikeLoop();

    float SynapticDecayLauncher();

    float MembranePotentialLauncher();

    Synapses *CreateDeviceSynapses( Synapses *syn );

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Trial data.
    double _elapsedTime;  // total time elapsed in a trial.
    int _spikeCounter;
    SynfireParameters _params;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Neuron Host Data
    Neuron *_network;
    float *_ranCache;
    bool *_spikeFlags;

    // Neuron Device Data
    Neuron *_dnetwork;
    float *_dranCache;
    bool *_dspikeFlags;

    // Neuron helpers.
    int network_size;
    std::vector<int> _whospiked;  // labels of Neurons who spiked during a timestep.
    matrix_t _spikeHistory;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Synapse Data
    Synapses _connectivity, *_dconnectivity;
    Synapses _inhibition_strength, *_dinh_str;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Training data
    int *_train_lab; // training labels.
    double *_train_times;
    double _train_freq; // training spike frequency (ms)^(-1)
    double _train_amp;  // training spike strength
    double _train_dur;  // training duration in ms
};

#endif //PORT_CUSYNFIRE_H

