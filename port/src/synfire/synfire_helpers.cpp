#include "synfire_helpers.h"

SynfireParameters::SynfireParameters() {
    SetDefaults();
}

SynfireParameters::SynfireParameters( int argc, char *argv[] ) {
    SetDefaults();

    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        cmd = argv[i];

        if (cmd[0] != '-') continue;
        switch (cmd[1]) {
            case 'L': // Synapses are static
                plasticity = false;
                break;

                //OUTPUT & ANALYSIS FLAGS
            case 'd': // '-d' outputs various statistics after each trial for diagnostic purposes (i.e. spk count, avg volt, etc)
                stats_on = 1;

                //~ PARAMETER FLAGS
            case 'A': // '-A <int>' sets the numbers of trials before termination of the run (default is 200000)
                trials = atoi(argv[++i]);
                break;

            case 'C': // '-C <float>' sets the number of ms per trial (default is 2000)
                trial_duration = atof(argv[++i]);
                break;

            case 'B': // '-B <int>' sets the number of training neurons (default is 10)
                ntrain = atoi(argv[++i]);
                break;

            case 'D': // '-D <int>' sets the max number of supersynapses a neuron may have
                tempNSS = atoi(argv[++i]);
                break;

            case 'c': // '-c <double>' sets the decay rate of the synapses (default is .999996)
                syndec = atof(argv[++i]);
                break;

            case 'f': // '-f <double>' sets the fraction of synapses initially active (default is .1)
                frac = atof(argv[++i]);
                if (frac >= 1 || frac < 0) {
                    std::cerr << "Command line input -f is invalid, must be <1 and >=0" << std::endl;
                    exit(1);
                }
                break;

            case 'g': // '-g <float>' sets GLTP
                eq_syn = atof(argv[++i]);
                break;

            case 'i': // '-i <double>' sets the amplitude of the global inhibition (default is .3)
                global_i = atof(argv[++i]);
                break;

            case 'j': // '-j <float>' sets the inhibition delay
                inh_d = atof(argv[++i]);
                break;

            case 'm': // '-m <double>' sets the excitatory spontaneous frequency (default is 40Hz)
                exfreq = atof(argv[++i]);
                break;

            case 'n':// '-n <double>' sets the inhibitory spontaneous frequency (default is 200Hz)
                infreq = atof(argv[++i]);
                break;

            case 'o':// '-o <double>' sets the amplitude of the spontaneous excitation (default is 1.3)
                examp = atof(argv[++i]);
                break;

            case 'p':// '-p <double>' sets the amplitude of the spontaneous inhibition (default is .1)
                inamp = atof(argv[++i]);
                break;

            case 'q':// '-q <double>' sets the active syn threshold (default is .2)
                act = atof(argv[++i]);
                break;

            case 'r':// '-r <double>' sets the super syn threshold (default is .4)
                sup = atof(argv[++i]);
                break;

            case 's':// '-s <double>' sets the synapse maximum (default is .6)
                cap = atof(argv[++i]);
                break;

            case 'u': // '-u <double>' set maximum inhibitory synaptic strength
                isynmax = atof(argv[++i]);
                break;

            case 'x': // '-x <double>' sets spike rate of training excitation (default is 1500 Hz)
                training_f = atof(argv[++i]) * 0.001;
                //convert to (ms)^(-1)
                break;

            case 'y': // '-y <double>' sets amplitude of training excitation (default is .7)
                training_amp = atof(argv[++i]);
                break;

            case 'z': // '-z <double>' sets the training excitation duration (default is 8 ms)
                training_t = atof(argv[++i]);
                break;

            default:
                std::cout << "Warning: command line flag " << cmd << " is not recognized." << std::endl;
                break;
        }
    }
}

void SynfireParameters::SetDefaults() {
    //~ Run Parameters
    timestep = 0.1;
    network_size = 200;
    trials = 200000;
    trial_duration = 2000;
    stats_on = true;

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
