#include <vector>
#include <iostream>
#include "random.h"
#include "microtime.h"
#include "synfire.h"
#include "neuron.h"

Synfire::Synfire( int nsize )
        : DT(0.1),
          INV_DT(1 / DT),
          trials(200000),
          trial_duration(2000),
          trial_steps((int) (trial_duration * INV_DT)),
          network_size(nsize),
          _connectivity(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn),
          _inhibition_strength(1, global_i, 1, 1, 1, 1, 2, network_size, tempNSS, window, eq_syn) {
    Initialize();
}

Synfire::Synfire( int nsize, double dt, int num_trials, int trial_time )
        : DT(dt),
          INV_DT(1 / dt),
          trials(num_trials),
          trial_duration(trial_time),
          trial_steps((int) (trial_time / dt)),
          _connectivity(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn),
          _inhibition_strength(1, global_i, 1, 1, 1, 1, 2, network_size, tempNSS, window, eq_syn) {
    network_size = nsize;

    Initialize();

}

void Synfire::Initialize() {
    // L1089: Seed random number generator & check RAND_MAX
    seed = time(NULL) * (-1);

    //==========================================
    //~ Initialize dependencies.
    NSS = network_size;
    group_rank = new int[network_size];

    if (man_tt == false) {
        train_times = new double[2];
        train_times[0] = 0.0;
        train_times[1] = trial_duration + 1000;
    }

    //==========================================
    //~ Initialize loaded dependencies
    group_s = network_size / ntrg;  // L1086

    // Training labels.
    train_lab = new int[ntrain * ntrg];
    for (int j = 0, tc = 0; j < ntrg; ++j) {
        for (int i = 0; i < ntrain; ++i, ++tc) {
            train_lab[tc] = j * group_s + i;
        }
    }

    // Inhibition delay
    dsteps = (int) (1 + INV_DT * inh_d);
    inh = new int[dsteps];
    for (int i = 0; i < dsteps; ++i) inh[i] = 0;

    //==========================================
    //~ Initialize Synapses.
    _connectivity = Synapses(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn);
    _inhibition_strength = Synapses(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn);

    //==========================================
    //~ Initialize Neurons.
    _network = new Neuron[network_size];
    for (int i = 0; i < network_size; ++i) {
        _network[i] = Neuron(i, exfreq, infreq, examp, inamp, global_i);
    }

}

void Synfire::Run() {
    double start, stop;

    // From L1230:
    double tTa[10], tTSa[trial_steps], tSDa[10];
    double tT[3], tTS[3], tMPL[3], tSL[3], tSD[3];
    tMPL[2] = 0, tSL[2] = 0, tTS[2] = 0, tSD[2] = 0;

    // L1234: for (int a=0; a<=10/*trials*/; a++){//****Trial Loop****//
    for (int a = 0; a <= 10; a++) {
        start = microtime();

        // Omitted: L1240 - 1263
        RunTrial(tT, tTS, tMPL, tSL, tTSa);

        // L1396: Reset Trial
        tSL[2] = 0;
        tMPL[2] = 0;
        t = 0.0;
        seed = time(NULL) * (-1);
        ran1(&seed);

        tSD[0] = microtime();
        if (plasticity) {
            // L1408: Synapses decay after each trial.
            _connectivity.SynapticDecay();
        }

        tSD[1] = microtime();
        tSDa[a] = (tSD[1] - tSD[0]);
        stop = microtime();

        if (stats_on) {
            std::cout << "Trial " << a << " took " << (stop - start) << " ms." << std::endl;
            sc = 0;
            av = 0;
        }

        // Reset neuron values for next trial
        for (int i = 0; i < network_size; ++i) {
            _network[i].Reset();
            _network[i].ResetSpike();
        }

        tT[1] = microtime();
        tT[2] = (tT[1] - tT[0]);
        tTa[a] = tT[2];
    }
}

void Synfire::RunTrial( double *tT, double *tTS, double *tMPL, double *tSL, double *tTSa ) {
    std::vector<int> v_whospike;

    // ref: L1235.
    tT[0] = microtime();
    int train_time_counter = 0;
    int train_group_lab = 0;

    tTS[0] = microtime();

    // L1268: for (int i=0; i<trial_steps; i++) {//****Timestep Loop****//
    for (int i = 0; i < trial_steps; i++) {
        // L1271: Training loop
        if (t >= train_times[train_time_counter]) {
            int tstart = ntrain * train_group_lab;
            int tstop = ntrain * (train_group_lab + 1);

            for (int j = tstart; j < tstop; ++j) {
                if (ran1(&seed) < training_f * DT) {
                    _network[train_lab[j]].ExciteInhibit(training_amp, 'e');
                    //cout<<"excited "<<train_lab[j]<<" at "<<t<<endl;
                }
            }

            // L1279: IDK yet.
            if (t >= (train_times[train_time_counter] + training_t)) {
                //cout<<train_group_lab<<" "<<t<<endl;
                train_time_counter++;
                train_group_lab = 1; // rand() % ntrg
            }
        }

        // L1286: Omitted Track voltages and conductances
        tMPL[0] = microtime();

        // L1295: Update membrane potentials first, keep track of who spikes
        for (int j = 0; j < network_size; ++j) {
            //if neuron j spikes this timestep, increase the length of v_whospike, store label
            if (_network[j].Update(DT)) {
                v_whospike.push_back(j);
            }
        }

        whocount = v_whospike.size();

        // L1333: Update tMPL
        tMPL[1] = microtime();
        tMPL[2] += (tMPL[1] - tMPL[0]);
        // cout << "MPL current: " << tMPL[1]-tMPL[0] << "MPL Total: " << tMPL[2];

        tSL[0] = microtime();
        for (int j = 0; j < v_whospike.size(); ++j) { // Spike Loop
            int spiker = whospike[j];
            int *send_to;//pointer to array containing post neurons
            int send_count;//number of post neurons receiving spike
//            cout<<v_whospike[j]<<" spikes!!! at "<<t<<endl;
//            _network[v_whospike[j]].recspike(t); //record time of spike
            sc++; //keep track of total number of spikes this trial

            // TODO: L1348: Emit spikes
            //check to see if spiking neuron is saturated
            if(_connectivity.GetPostSynapticLabel('s', spiker, send_to) == _connectivity.GetNSS(spiker)) {
                int j_nss = _connectivity.GetNSS(spiker);
                for(int k = 0; k < j_nss; ++k) {
                    // Send spikes along super synapses
                    _network[send_to[k]].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, send_to[k]), 'e');
                }
            } else {
                //spiking neuron isn't saturated, send spikes along active connections
                send_count = _connectivity.GetPostSynapticLabel('a', spiker, send_to);
                for(int k = 0; k < send_count; k++) {
                    _network[send_to[k]].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, send_to[k]), 'e');
                }
            }

            if(plasticity) {
                _connectivity.Synaptic_Plasticity(spiker, t, &_network);
            }
        }

        tSL[1] = microtime();
        tSL[2] += (tSL[1] - tSL[0]);
        // cout << "SL current: " << tSL[1]-tSL[0] << " SL total:" << tSL[2] << endl;

        // L1372: Inhibition.
        inh[dsteps - 1] = whocount;
        for (int z = 0; z < whocount; ++z) {
            int spiker = whospike[z];
            for (int j = 0; j < network_size; ++j) {
                _network[j].ExciteInhibit(_inhibition_strength.GetSynapticStrength(spiker, j), 'i');
            }
//            cout<<t<<" "<<inh[0]*global_i<<endl;
        }

        for (int y = 0; y < dsteps - 1; ++y) {
            inh[y] = inh[y + 1];
        }

        // L1388: Resets the stored spikes.
        if (whocount != 0) {
            v_whospike.clear();
            delete[] whospike; //reset whocount and destroy whospike at end of timestep
            whocount = 0;
        }

        t += DT; // increment timer.
        tTS[1] = microtime();
        tTS[1] = microtime();
        tTSa[i] = (tTS[1] - tTS[0]);
    } // end timestep loop.
}