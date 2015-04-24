#include <vector>
#include <iostream>
#include "random.h"
#include "microtime.h"
#include "synfire.h"
#include "neuron.h"

//"""""""""""""""""""""""""""""""""""""""""""""""""
// TODO: Pending attributes from the top of synfireGrowth.cpp.
int *group_rank; //contains group ranking of each neuron after chain network forms

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

Synfire Synfire::CreateSynfire() {
    return Synfire(SynfireParameters());
}

Synfire Synfire::CreateSynfire( int nsize ) {
    struct SynfireParameters parms;
    parms.network_size = nsize;
    return Synfire(parms);
}

Synfire Synfire::CreateSynfire( int nsize, double dt, int num_trials, int trial_time ) {
    struct SynfireParameters parms;
    parms.network_size = nsize;
    parms.timestep = dt;
    parms.trials = num_trials;
    parms.trial_duration = trial_time;
    return Synfire(parms);
}

Synfire::Synfire( SynfireParameters params )
        : DT(params.timestep),
          INV_DT(1 / DT),
          trials(params.timestep),
          trial_duration(params.trial_duration),
          trial_steps((int) (trial_duration * INV_DT)),
          network_size(params.network_size),
          _connectivity(params.frac,
                        0.0,
                        params.act,
                        params.sup,
                        params.cap,
                        params.syndec,
                        params.conn_type,
                        network_size,
                        params.tempNSS,
                        params.window,
                        params.eq_syn),
          _inhibition_strength(1,
                               params.global_i,
                               1, 1, 1, 1, 2,
                               network_size,
                               params.tempNSS,
                               params.window,
                               params.eq_syn) {
    _params = params;
    Initialize();
}

void Synfire::Initialize() {
    //~ Ensure parameter sanity.
    _elapsedTime = 0.0;
    _spikeCounter = 0;

    // L1089: Seed random number generator & check RAND_MAX
    seed = -1 * time(NULL);

    //==========================================
    //~ Statistics & Logging.
    stats_on = true;
    stats_av = 0;

    //==========================================
    //~ Initialize dependencies.
    group_rank = new int[network_size];

    if (_params.man_tt == false) {
        _train_times = new double[2];
        _train_times[0] = 0.0;
        _train_times[1] = trial_duration + 1000;
    }

    //==========================================
    //~ Initialize loaded dependencies
    group_s = network_size / _params.ntrg;  // L1086

    // Training labels.
    _train_lab = new int[_params.ntrain * _params.ntrg];
    for (int j = 0, tc = 0; j < _params.ntrg; ++j) {
        for (int i = 0; i < _params.ntrain; ++i, ++tc) {
            _train_lab[tc] = j * group_s + i;
        }
    }

    // Inhibition delay
    dsteps = (int) (1 + INV_DT * _params.inh_d);
    inh = new int[dsteps];
    for (int i = 0; i < dsteps; ++i) inh[i] = 0;

    //==========================================
    //~ Initialize Synapses.
//    _connectivity = Synapses(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn);
//    _inhibition_strength = Synapses(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn);

    //==========================================
    //~ Initialize Neurons & their helpers
    _network = new Neuron[network_size];
    for (int i = 0; i < network_size; ++i) {
        _network[i] = Neuron(i, _params.exfreq, _params.infreq, _params.examp, _params.inamp, _params.global_i);
    }

    _whospiked.reserve((unsigned long) (network_size / 2));
    _spikeHistory.resize((unsigned long) (network_size), row_t());
}

void Synfire::Run() {
    double start, stop;

    // From L1230:
    double tTa[10], tTSa[trial_steps], tSDa[10];
    double tT[3], tTS[3], tMPL[3], tSL[3], tSynDecay[3];
    tMPL[2] = 0, tSL[2] = 0, tTS[2] = 0, tSynDecay[2] = 0;

    // L1234: for (int a=0; a<=10/*trials*/; a++){//****Trial Loop****//
    for (int a = 0; a <= 10; a++) {
        start = microtime();

        // Omitted: L1240 - 1263
        RunTrial(tT, tTS, tMPL, tSL, tTSa);

        // L1396: Reset Trial
        tSL[2] = 0;
        tMPL[2] = 0;
        _elapsedTime = 0.0;

        seed = -1 * time(NULL);
        ran1(&seed);

        tSynDecay[0] = microtime();
        if (_params.plasticity) { // L1408: Synapses decay after each trial.
            _connectivity.SynapticDecay();
        }
        tSynDecay[1] = microtime();

        tSDa[a] = (tSynDecay[2] = (tSynDecay[1] - tSynDecay[0]));
        stop = microtime();

        if (stats_on) {
            stats_av = 0.0;
            for (int i = 0; i < network_size; ++i) stats_av += _network[i].Volts();
            stats_av /= network_size;

            // FORMAT: <trial> <spike total> <av. volt> <runtime> <# of active connections>
            std::cout << "Trial " << a << std::endl;
            std::cout << "\tDuration: " << (stop - start) << " ms." << std::endl;
        }

        // Reset neuron values for next trial
        for (int i = 0; i < network_size; ++i) {
            _network[i].Reset();
            _network[i].ResetSpike();
        }

        _spikeCounter = 0;

        // Update timing data.
        tT[1] = microtime();
        tT[2] = (tT[1] - tT[0]);
        tTa[a] = tT[2];
    }
}

/**
 * Runs a single SynfireGrowth trial.
 * TODO: Figure out and rename tT, tTS, tMPL, tSL, tTSa.
 *
 * @param tT   idk?
 * @param tTS  idk?
 * @param tMPL The timings for Membrane Potential Layer, e.g. [start, end, delta]. (?)
 * @param tSL  The timings for the Spike loop, e.g. [start, end, delta]. (?)
 * @param tTSa [description]
 */
double Synfire::RunTrial( double *tT, double *tTS, double *tMPL, double *tSpkLp, double *tTSa ) {
    // ref: L1235.
    tT[0] = microtime();
    tTS[0] = microtime();

    int train_time_counter = 0;
    int train_group_lab = 0;
    _elapsedTime = 0.0;
    _spikeCounter = 0;

    // L1268: for (int i=0; i<trial_steps; i++) {//****Timestep Loop****//
    for (int i = 0; i < trial_steps; ++i) {
        //---------------------------------
        // L1271: Training loop
        if (_elapsedTime >= _train_times[train_time_counter]) {
            int tstart = _params.ntrain * train_group_lab;
            int tstop = _params.ntrain * (train_group_lab + 1); // TODO: isnt this just tstart + ntrain?
            double tfreq_thresh = _train_freq * DT;

            for (int j = tstart; j < tstop; ++j) {
                if (ran1(&seed) < tfreq_thresh) {
                    _network[_train_lab[j]].ExciteInhibit(_train_amp, 'e');
                    //std::cout << "excited " << train_lab[j] << " at " << t << std::endl;
                }

                // L1279: IDK yet.
                if (_elapsedTime >= (_train_times[train_time_counter] + _train_dur)) {
                    //std::cout << train_group_lab << " " << t << std::endl;
                    train_time_counter++;
                    train_group_lab = 1; // rand() % ntrg
                }
            }
        }

        //---------------------------------
        // L1286: Omitted Track voltages and conductances
        tMPL[0] = microtime();

        //---------------------------------
        // Enter Membrane Potential Layer loop.
        // L1295: Update membrane potentials first, keep track of who spikes
        for (int j = 0; j < network_size; ++j) {
            if (_network[j].Update(DT)) {
                _whospiked.push_back(j);
            }
        }

        // L1333: Update tMPL
        tMPL[1] = microtime();
        tMPL[2] += (tMPL[1] - tMPL[0]);
        // std::cout << "MPL current: " << tMPL[1]-tMPL[0] << "MPL Total: " << tMPL[2] << std::endl;

        //---------------------------------
        // Enter Spike Loop
        tSpkLp[0] = microtime();
        DoSpikeLoop();
        tSpkLp[1] = microtime();
        tSpkLp[2] += (tSpkLp[1] - tSpkLp[0]);


        //---------------------------------
        // Inhibition
        inh[dsteps - 1] = _whospiked.size();
        for (std::vector<int>::iterator itr = _whospiked.begin(); itr != _whospiked.end(); ++itr) {
            for (int j = 0; j < network_size; ++j) {
                _network[j].ExciteInhibit(_inhibition_strength.GetSynapticStrength((*itr), j), 'i');
            }
        }

        for (int j = 0; j < dsteps - 1; ++j) {
            inh[j] = inh[j + 1];
        }

        // L1388: Reset spikes for this timestep.
        _whospiked.clear();

        // Prepare timing for next timestep.
        _elapsedTime += DT;
        tTS[1] = microtime();
        tTSa[i] = (tTS[1] - tTS[0]);
    }

    return _elapsedTime;
}

void Synfire::DoSpikeLoop() {
    int spiker;
    row_t spk_hist; // Neuron's spike history data.
    bool *send_to;  // pointer to array containing post neurons.
    int send_count; // number of post neurons receiving spike.

    for (std::vector<int>::iterator itr = _whospiked.begin(); itr != _whospiked.end(); ++itr) {
        spiker = (*itr);
        spk_hist = _spikeHistory[spiker];

        // Log the spike event.
        spk_hist.push_back(_elapsedTime);
        ++_spikeCounter;
//        std::cout << spiker <<" spikes!!! at " << _elapsedTime << std::endl;

        // L1348: Emit spikes
        // Check to see if spiking neuron is saturated
        if (_connectivity.GetPostSynapticLabel('s', spiker, send_to) == _connectivity.GetNSS(spiker)) {
            int j_nss = _connectivity.GetNSS(spiker);

            // TODO: can't just loop over send_to anymore.
            for (int k = 0; k < j_nss; ++k) { // Send spikes along super synapses
                _network[send_to[k]].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, send_to[k]), 'e');
            }
        } else { // Spiking neuron isn't saturated, send spikes along active connections
            send_count = _connectivity.GetPostSynapticLabel('a', spiker, send_to);

            // TODO: can't just loop over send_to anymore.
            for (int k = 0; k < send_count; k++) {
                _network[send_to[k]].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, send_to[k]), 'e');
            }
        }

        if (_params.plasticity) {
            _connectivity.Synaptic_Plasticity(spiker, _elapsedTime, &spk_hist[0], spk_hist.size());
        }
    }
}

double Synfire::GetAverageVoltage() {
    double avg = 0.0;

    for (int i = 0; i < network_size; ++i) {
        avg += _network[i].Volts();
    }

    return (avg / network_size);
}
