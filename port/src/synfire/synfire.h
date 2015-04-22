#ifndef PORT_SYNFIRE_H
#define PORT_SYNFIRE_H

#include <fstream>

//~ Forward declarations of dependencies
class Neuron;

class Synfire {
  public:

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Constructors.
    //"""""""""""""""""""""""""""""""""""""""""""""""""

    Synfire( int nsize ) : DT(0.1),
                           INV_DT(1 / DT),
                           trials(200000),
                           trial_duration(2000),
                           trial_steps((int) (trial_duration * INV_DT)) {
        Initialize();
    }

    Synfire( int nsize, double dt, int num_trials, int trial_time );

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Synfire Methods
    //"""""""""""""""""""""""""""""""""""""""""""""""""
    void Run();

    void RunTrial( double *tT, double *tTS, double *tMPL, double *tSL, double *tTSa );

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Member constants.
    const double DT, INV_DT;
    const int trials, trial_duration, trial_steps;
    int *train_lab; // training labels.

    // Inhibition delay data.
    int dsteps;
    int *inh;

    // TODO: Not yet understood
    int group_s;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // TODO: Pending attributes from the top of synfireGrowth.cpp.
    int runid = 1;  //identifies the particular run
    bool LOAD = false; //1 means load data
    bool ILOAD = false;

    //Tracking variables during trial
    std::ofstream track_volt; //stream that tracks volt and conds during trial
    int volt_post, volt_pre = 0; //contains the label of the neuron whose voltage is being tracked, it postsynaptic to volt_pre

    double t = 0.0; //current time in ms
    int *whospike, whocount = 0;  //tracks labels of neurons that spike during current step, length of whospike[]
    int *group_rank; //contains group ranking of each neuron after chain network forms

    //Spontaneous activity defaults
    double exfreq = 40, infreq = 200, examp = 1.3, inamp = .1, global_i = .3, inh_d = 0, leak = -85.0;

    //Synapses defaults
    int NSS, tempNSS = 10; //max # of supersynapses allowed
    double act = .2, sup = .4, cap = .6, frac = .1, isynmax = .3, eq_syn = .3;
    double syndec = .99999;
    int conn_type = 1;
    bool plasticity = true;
    double window = 200; //history time window size (ms)

    //Training defaults
    int ntrain = 10, ntrg = 1; //number of training neurons
    bool man_tt = false; //manual training time bool (false if training occurs at t=0)
    double *train_times;
    double training_f = 1.5; //training spike frequency (ms)^(-1)
    double training_amp = .7; //training spike strength
    double training_t = 8.0; //training duration in ms


    // Stats
    bool stats_on = true;
    int sc = 0; //total spike counter
    double av = 0;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // TODO: Pending attributes from synfireGrowth.main()

  private:
    void Initialize();

    Neuron *_network;
    int network_size;
};

#endif //PORT_SYNFIRE_H
