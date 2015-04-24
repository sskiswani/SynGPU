#ifndef PORT_SYNFIRE_H
#define PORT_SYNFIRE_H

#include <vector>
#include <fstream>
#include "synapses.h"

//~ Forward declarations of dependencies
class Neuron;

class Synfire {
  public:

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Constructors.
    //"""""""""""""""""""""""""""""""""""""""""""""""""

    Synfire( int nsize );

    Synfire( int nsize, double dt, int num_trials, int trial_time );

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Synfire Methods
    //"""""""""""""""""""""""""""""""""""""""""""""""""
    void Run();

    void RunTrialRobust( double *tT, double *tTS, double *tMPL, double *tSL, double *tTSa );

    void RunTrial( double *tT, double *tTS, double *tMPL, double *tSL, double *tTSa );

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Member constants.
    const double DT, INV_DT;
    const int trials, trial_duration, trial_steps;
    int *train_lab; // training labels.

    //~ Inhibition delay data.
    int dsteps;
    int *inh;

    std::vector<int> _whospiked;  // labels of Neurons who spiked during a timestep.

    // TODO: Not yet understood
    int group_s;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // TODO: Pending attributes from synfireGrowth.main()

  private:
    void Initialize();

    Synapses _connectivity;
    Synapses _inhibition_strength;
    Neuron *_network;
    int network_size;

    //~ Timestep info
    double _elapsed;  // total time elapsed in a trial.
};

#endif //PORT_SYNFIRE_H
