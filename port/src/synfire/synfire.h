#ifndef PORT_SYNFIRE_H
#define PORT_SYNFIRE_H

#include <vector>
#include <fstream>
#include "synapses.h"

typedef std::vector<double> row_t;
typedef std::vector<row_t> matrix_t;

//~ Forward declarations of dependencies
class Neuron;

struct SynfireParameters {
    SynfireParameters();

    int network_size;
//    bool LOAD = false; //1 means load data
//    bool ILOAD = false;

    //~ Trial parameters.
    int trials;             // default # of trials
    double timestep;        // timestep (ms)
    double trial_duration;  // length of trial (ms)

    //~ Save strings
//    string loadpath, iloadpath, synapse_path = "syn/syn", roster_path = "roster/roster", volt_e_path = "volt/volt_ensemble";
//    string volt_t_path = "volt/volt_track_", conn_path = "connect/connect", r_conn_path = "read/net_struct";
//    string dat_suff = ".dat", txt_suff = ".txt", sim_base = ".", sim_lab;
//    int synapse_save = 100, roster_save = 1000, volt_save = 1000, conn_save = 1000; //save data intervals

    //~ Tracking variables during trial
//    ofstream track_volt; // stream that tracks volt and conds during trial
//    int volt_post, volt_pre = 0; // contains the label of the neuron whose voltage is being tracked, it postsynaptic to volt_pre

    //~ Spontaneous activity defaults
    double exfreq, infreq, examp, inamp, global_i, inh_d, leak;

    // Synapses defaults
    int NSS, tempNSS;   // max # of supersynapses allowed.
    double act, sup, cap, frac, isynmax, eq_syn;
    double syndec;
    int conn_type;
    bool plasticity;
    double window;      // history time window size (ms)

    //~ Training defaults
    int ntrain, ntrg;   // number of training neurons
    bool man_tt;        // manual training time bool (false if training occurs at t=0)
    double training_f;      // training spike frequency (ms)^(-1)
    double training_amp;    // training spike strength
    double training_t;      // training duration in ms
};

class Synfire {
  public:
    static Synfire CreateSynfire();

    static Synfire CreateSynfire( int nsize );

    static Synfire CreateSynfire( int nsize, double dt, int num_trials, int trial_time );

    //~ CTOR
    Synfire( SynfireParameters );

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

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Helpers & Accessors
    double GetAverageVoltage();


    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Member constants.
    const double DT, INV_DT;
    const int trials, trial_duration, trial_steps;

    bool stats_on;
    double stats_av;

    //~ Inhibition delay data.
    int dsteps;
    int *inh;


    // TODO: Not yet understood
    int group_s;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // TODO: Pending attributes from synfireGrowth.main()

  private:
    void Initialize();

    void DoSpikeLoop();

    SynfireParameters _params;

    //~ Neuron Data
    int network_size;
    Neuron *_network;

    //~ Neuron helpers
    std::vector<int> _whospiked;  // labels of Neurons who spiked during a timestep.
    matrix_t _spikeHistory;

    //~ Synapse Data
    Synapses _connectivity;
    Synapses _inhibition_strength;

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

#endif //PORT_SYNFIRE_H
