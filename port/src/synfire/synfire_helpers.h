#ifndef PORT_SYNFIRE_HELPERS_H
#define PORT_SYNFIRE_HELPERS_H

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

#endif //PORT_SYNFIRE_HELPERS_H
