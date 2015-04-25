#ifndef PORT_SYNFIRE_H
#define PORT_SYNFIRE_H

#include <vector>
#include <fstream>
#include "synfire_helpers.h"
#include "synapses.h"

typedef std::vector<double> row_t;
typedef std::vector<row_t> matrix_t;

class Neuron;

class Synfire {
  public:
    static Synfire CreateSynfire();

    static Synfire CreateSynfire( int nsize );

    static Synfire CreateSynfire( int nsize, double dt, int num_trials, double trial_time );

    // ctor
    Synfire( SynfireParameters );

    // dtor.
    ~Synfire();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Synfire Methods
    //"""""""""""""""""""""""""""""""""""""""""""""""""

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

    // TODO: Not yet understood
    int group_s;

  private:
    void Initialize();

    void DoSpikeLoop();


    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Neuron Data
    int network_size;
    Neuron *_network;
    std::vector<int> _whospiked;  // labels of Neurons who spiked during a timestep.
    matrix_t _spikeHistory;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Synapse Data
    Synapses _connectivity;
    Synapses _inhibition_strength;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Trial data.
    double _elapsedTime;  // total time elapsed in a trial.
    int _spikeCounter;
    SynfireParameters _params;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Training data
    int *_train_lab; // training labels.
    double *_train_times;
    double _train_freq; // training spike frequency (ms)^(-1)
    double _train_amp;  // training spike strength
    double _train_dur;  // training duration in ms
};

#endif //PORT_SYNFIRE_H
