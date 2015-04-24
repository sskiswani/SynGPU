#include <vector>
#include <iostream>
#include <time.h>
#include "../common/random.h"
#include "../common/microtime.h"
#include "synfire_helpers.h"
#include "synfire.h"
#include "neuron.h"

//"""""""""""""""""""""""""""""""""""""""""""""""""
// TODO: Pending attributes from the top of synfireGrowth.cpp.
int *group_rank; //contains group ranking of each neuron after chain network forms

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
	double tTa[10];
	double tSDa[10];
	double *tTSa = new double[trial_steps];
    double tT[3], tTS[3], tMPL[3], tSL[3], tSynDecay[3];
    tMPL[2] = 0, tSL[2] = 0, tTS[2] = 0, tSynDecay[2] = 0;

    // L1234: for (int a=0; a<=10/*trials*/; a++){//****Trial Loop****//
    // TODO: Loop for the appropriate number of trials.
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
            double avg_volts = 0.0;
            for (int i = 0; i < network_size; ++i) avg_volts += _network[i].Volts();
            avg_volts /= network_size;

            std::cout << "Trial " << a << std::endl;
            std::cout << "\tDuration: " << US_TO_MS(stop - start) << " ms." << std::endl;
            std::cout << "\tSpikes: " << _spikeCounter << std::endl;
            std::cout << "\tAvg. Volts: " << avg_volts << std::endl;
            std::cout << "\tActive Connections: " << _connectivity.CountSynapses('a') << std::endl;

            // Do error checking.
            bool *send_to;
            int send_count, act_count;
            for(int i = 0; i < network_size; ++i) {
                send_count = _connectivity.GetPostSynapticLabel('a', i, send_to);
                act_count = 0;
                for(int j = 0; j < network_size; ++j) {
                    if(send_to[j] == true) ++act_count;
                }

                if(act_count != send_count) {
                    std::cout << "Neuron[" << i << "] reports " << send_count << " connections but actually has " << act_count << " connections." << std::endl;
                }
            }
        }

        // Reset neuron values for next trial
        _spikeCounter = 0;
        for (matrix_t::iterator itr = _spikeHistory.begin(); itr != _spikeHistory.end(); ++itr) {
            (*itr).clear();
        }

        for (int i = 0; i < network_size; ++i) {
            _network[i].Reset();
        }

        // Update timing data.
        tT[1] = microtime();
        tT[2] = (tT[1] - tT[0]);
        tTa[a] = tT[2];
    } // end of simulation.

    if (stats_on == false) return;

    std::cout << "\n\nSIMULATION SUMMARY:" << std::endl;
    double avgTime = 0.0, avgTS = 0.0, avgSD = 0;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Trial: " << i << " Time: " << US_TO_MS(tTa[i]) << " ms TrialStep: " << US_TO_MS(tTSa[i]) <<
        " SynapticDecay: " << US_TO_MS(tSDa[i]) << std::endl;

        avgTime += tTa[i];
        avgTS += tTSa[i];
        avgSD += tSDa[i];
    }

    std::cout << "\nTIMING:\n";
    std::cout << "Avg Trial: " << US_TO_MS(avgTime / 10) << " Avg TrialStep: " << US_TO_MS(avgTS / 10) <<
    " Avg Decay: " << US_TO_MS(avgSD / 10) << std::endl;
}

/**
 * Runs a single SynfireGrowth trial.
 *
 * @param tT   The timing for the current trial.
 * @param tTS  The timing for an individual trial step.
 * @param tMPL The timing for Membrane Potential Layer, e.g. [start, end, delta]. (?)
 * @param tSL  The timing for the Spike loop, e.g. [start, end, delta]. (?)
 * @param tTSa The timing across all trial steps.
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
    int send_count;

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
            int j_nss = _connectivity.GetNSS(spiker); // TODO: Figure out NSS indexing with send_to.

            for(int post = 0, j = 0; post < network_size && j < j_nss; ++post) {
                if(send_to[post] == false) continue;
                _network[post].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, post), 'e');
                ++j;
            }
        } else { // Spiking neuron isn't saturated, send spikes along active connections
            send_count = _connectivity.GetPostSynapticLabel('a', spiker, send_to);

            for (int post = 0, j = 0; post < network_size && j < send_count; post++) {
                if(send_to[post] == false) continue;
                _network[post].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, post), 'e');
                ++j;
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
