#include "synfire_helpers.h"

SynfireParameters::SynfireParameters() {
    //~ Run Parameters
    timestep = 0.1;
    network_size = 200;
    trials = 200000;
    trial_duration = 2000;

    //~ Spontaneous activity defaults.
    exfreq = 40;
    infreq = 200;
    examp = 1.3;
    inamp = .1;
    global_i = .3;
    inh_d = 0;
    leak = -85.0;

    //~ Synapses defaults
    NSS = network_size;
    tempNSS = 10;

    act = 0.2;
    sup = 0.4;
    cap = 0.6;
    frac = 0.1;
    isynmax = 0.3;
    eq_syn = 0.3;

    syndec = 0.99999;
    conn_type = 1;
    plasticity = true;
    window = 200;

    //~ Training defaults
    ntrain = 10;
    ntrg = 1;
    man_tt = false;
    training_f = 1.5;
    training_amp = 0.7;
    training_t = 8.0;
}
