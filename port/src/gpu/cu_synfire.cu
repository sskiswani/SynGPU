#include <iostream>
#include <vector>
#include "cu_synfire.cuh"
#include "random.h"
#include "utility.h"
#include "helpers.h"
#include "neuron.cpp"
#include "synapses.cpp"
#include "cuda_utils.h"
#include "microtime.h"


CUSynfire CUSynfire::CreateCUSynfire() {
    return CUSynfire(SynfireParameters());
}

CUSynfire CUSynfire::CreateCUSynfire( int nsize ) {
    struct SynfireParameters parms;
    parms.network_size = nsize;
    return CUSynfire(parms);
}

CUSynfire CUSynfire::CreateCUSynfire( int nsize, double dt, int num_trials, double trial_time ) {
    struct SynfireParameters parms;
    parms.network_size = nsize;
    parms.timestep = dt;
    parms.trials = num_trials;
    parms.trial_duration = trial_time;
    return CUSynfire(parms);
}

CUSynfire::CUSynfire( SynfireParameters params )
        : DT(params.timestep),
          INV_DT(1 / DT),
          trials(params.timestep),
          trial_duration(params.trial_duration),
          trial_steps(trial_duration * INV_DT),
          stats_on(params.stats_on),
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

CUSynfire::~CUSynfire() {
    //~ Clear host data.
    delete [] _network;
    delete [] _train_lab;

    HANDLE_ERROR(cudaFreeHost(_spikeFlags));
    HANDLE_ERROR(cudaFree(_dspikeFlags));
    HANDLE_ERROR(cudaFreeHost(_ranCache));
    HANDLE_ERROR(cudaFree(_dranCache));

    HANDLE_ERROR(cudaFree(_dconnectivity));
    HANDLE_ERROR(cudaFree(_dinh_str));

    cudaDeviceSynchronize();
}

void CUSynfire::Initialize() {
    //~ Ensure parameter sanity.
    _elapsedTime = 0.0;
    _spikeCounter = 0;

    // L1089: Seed random number generator & check RAND_MAX
    seed = -1 * time(NULL);
    group_s = network_size / _params.ntrg;  // L1086


    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Initialize training data.
    if (_params.man_tt == false) {
        _train_times = new double[2];
        _train_times[0] = 0.0;
        _train_times[1] = trial_duration + 1000;
    }

    // Training labels.
    _train_lab = new int[_params.ntrain * _params.ntrg];
    for (int j = 0, tc = 0; j < _params.ntrg; ++j) {
        for (int i = 0; i < _params.ntrain; ++i, ++tc) {
            _train_lab[tc] = j * group_s + i;
        }
    }


    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Inhibition delay
    dsteps = (int) (1 + INV_DT * _params.inh_d);
    inh = new int[dsteps];
    for (int i = 0; i < dsteps; ++i) inh[i] = 0;


    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Initialize Neurons & their helpers
    _network = new Neuron[network_size];
    for (int i = 0; i < network_size; ++i) {
        _network[i] = Neuron(i, _params.exfreq, _params.infreq, _params.examp, _params.inamp, _params.global_i);
    }

    _whospiked.reserve((unsigned long) (network_size / 2));
    _spikeHistory.resize((unsigned long) (network_size), row_t());

    //~ Device related Neuron data.
    HANDLE_ERROR(cudaMalloc((void **) &_dnetwork, sizeof(Neuron) * network_size));
    HANDLE_ERROR(cudaMemcpy(_dnetwork, _network, sizeof(Neuron) * network_size, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMallocHost((void **) &_spikeFlags, sizeof(bool) * network_size));
    HANDLE_ERROR(cudaMalloc((void **) &_dspikeFlags, sizeof(bool) * network_size));

    HANDLE_ERROR(cudaMallocHost((void **) &_ranCache, sizeof(float) * network_size * 4));
    HANDLE_ERROR(cudaMalloc((void **) &_dranCache, sizeof(float) * network_size * 4));


    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Initialize Synapses & their helpers
    _dconnectivity = CreateDeviceSynapses(&_connectivity);
    _dinh_str = CreateDeviceSynapses(&_inhibition_strength);
}

void CUSynfire::Run() {
    Timer trialTimer;

    // From L1230:
    double tTa[10], tTSa[trial_steps], tSDa[10];
    double tT[3], tTS[3], tMPL[3], tSL[3], tSynDecay[3];
    tMPL[2] = 0, tSL[2] = 0, tTS[2] = 0, tSynDecay[2] = 0;

    // L1234: for (int a=0; a<=10/*trials*/; a++){//****Trial Loop****//
    // TODO: Loop for the appropriate number of trials.
    for (int a = 0; a <= 10; a++) {
        trialTimer.Start();

        LOG("Trial[%i] -- BEGIN", a)
        // Omitted: L1240 - 1263
        RunTrial(tT, tTS, tMPL, tSL, tTSa);

        LOG("Trial[%i] -- END", a)

        //"""""""""""""""""""""""""""""""""""""""""""""""""
        // L1396: Reset Trial
        tSL[2] = 0;
        tMPL[2] = 0;
        _elapsedTime = 0.0;

        seed = -1 * time(NULL);
        ran1(&seed);

        //"""""""""""""""""""""""""""""""""""""""""""""""""
        // Synaptic Decay
        tSynDecay[0] = microtime();
        if (_params.plasticity) { // L1408: Synapses decay after each trial.
            SynapticDecayLauncher();
        }
        tSynDecay[1] = microtime();

        tSDa[a] = (tSynDecay[2] = (tSynDecay[1] - tSynDecay[0]));
        trialTimer.Stop();

        LOG("Publishing stats.")
        if (stats_on) {
            double avg_volts = 0.0;
            for (int i = 0; i < network_size; ++i) avg_volts += _network[i].Volts();
            avg_volts /= network_size;

            std::cout << "Trial " << a << std::endl;
            std::cout << "\tDuration: " << US_TO_MS(trialTimer.Duration()) << " ms." << std::endl;
            std::cout << "\tSpikes: " << _spikeCounter << std::endl;
            std::cout << "\tAvg. Volts: " << avg_volts << std::endl;
            std::cout << "\tActive Connections: " << _connectivity.CountSynapses('a') << std::endl;

            // Do error checking.
            bool *send_to;
            int send_count, act_count;
            for (int i = 0; i < network_size; ++i) {
                send_count = _connectivity.GetPostSynapticLabel('a', i, send_to);
                act_count = 0;
                for (int j = 0; j < network_size; ++j) {
                    if (send_to[j] == true) ++act_count;
                }

                if (act_count != send_count) {
                    std::cout << "Neuron[" << i << "] reports " << send_count << " connections but actually has " <<
                    act_count << " connections." << std::endl;
                }
            }
        }

        //"""""""""""""""""""""""""""""""""""""""""""""""""
        // Reset and prep for next trial
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

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Print statistics
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
double CUSynfire::RunTrial( double *tT, double *tTS, double *tMPL, double *tSpkLp, double *tTSa ) {
    // ref: L1235.
    tT[0] = microtime();
    tTS[0] = microtime();

    int train_time_counter = 0;
    int train_group_lab = 0;
    _elapsedTime = 0.0;
    _spikeCounter = 0;

    // L1268: for (int i=0; i<trial_steps; i++) {//****Timestep Loop****//
    for (int i = 0; i < trial_steps; ++i) {

        //"""""""""""""""""""""""""""""""""""""""""""""""""
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

        //"""""""""""""""""""""""""""""""""""""""""""""""""
        // Enter Membrane Potential Layer loop.
        // L1286: Omitted Track voltages and conductances
        tMPL[0] = microtime();

        // L1295: Update membrane potentials first, keep track of who spikes
        MembranePotentialLauncher();

        // L1333: Update tMPL
        tMPL[1] = microtime();
        tMPL[2] += (tMPL[1] - tMPL[0]);
        // std::cout << "MPL current: " << tMPL[1]-tMPL[0] << "MPL Total: " << tMPL[2] << std::endl;

        //"""""""""""""""""""""""""""""""""""""""""""""""""
        // Enter Spike Loop
        tSpkLp[0] = microtime();
        DoSpikeLoop();
        tSpkLp[1] = microtime();
        tSpkLp[2] += (tSpkLp[1] - tSpkLp[0]);


        //"""""""""""""""""""""""""""""""""""""""""""""""""
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


void CUSynfire::DoSpikeLoop() {
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

        // L1348: Emit spikes
        // Check to see if spiking neuron is saturated
        if (_connectivity.GetPostSynapticLabel('s', spiker, send_to) == _connectivity.GetNSS(spiker)) {
            int j_nss = _connectivity.GetNSS(spiker); // TODO: Figure out NSS indexing with send_to.

            for (int post = 0, j = 0; post < network_size && j < j_nss; ++post) {
                if (send_to[post] == false) continue;
                _network[post].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, post), 'e');
                ++j;
            }
        } else { // Spiking neuron isn't saturated, send spikes along active connections
            send_count = _connectivity.GetPostSynapticLabel('a', spiker, send_to);

            for (int post = 0, j = 0; post < network_size && j < send_count; post++) {
                if (send_to[post] == false) continue;
                _network[post].ExciteInhibit(_connectivity.GetSynapticStrength(spiker, post), 'e');
                ++j;
            }
        }

        if (_params.plasticity) {
            _connectivity.Synaptic_Plasticity(spiker, _elapsedTime, &spk_hist[0], spk_hist.size());
        }
    }
}

double CUSynfire::GetAverageVoltage() {
    double avg = 0.0;

    for (int i = 0; i < network_size; ++i) {
        avg += _network[i].Volts();
    }

    return (avg / network_size);
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"

float CUSynfire::SynapticDecayLauncher() {
    LOG("Preparing Synaptic Decay Kernel")
    int numThreads = 256;
    int numBlocks = MIN( 32, (network_size + numThreads - 1) / numThreads );
    if(numBlocks <= 0) numBlocks = 1;
    if (network_size % numThreads == 0) ++numBlocks;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Start timers
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Launch kernel
    LOG("Launching Synaptic Decay Kernel")
    cudaDeviceSynchronize();
    SynapticDecayKernel<<<numBlocks, numThreads>>>(_dconnectivity, network_size);
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Cache references
    int *d_actcount = _connectivity._actcount;
    int *d_supcount = _connectivity._supcount;
    int *d_NSS = _connectivity._NSS;
    double *dev_G_data = _connectivity._G._data;
    bool *dev_supsyn_data = _connectivity._supsyn._data;
    bool *dev_actsyn_data = _connectivity._actsyn._data;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Copy over Synapse data.
    HANDLE_ERROR(cudaMemcpy(&_connectivity, _dconnectivity, sizeof(Synapses), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(d_supcount, _connectivity._supcount, sizeof(int) * network_size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(d_actcount, _connectivity._actcount, sizeof(int) * network_size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(d_NSS, _connectivity._NSS, sizeof(int) * network_size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    //~ Copy over TArray2 data.
    HANDLE_ERROR(cudaMemcpy(dev_G_data, _connectivity._G._data, _connectivity._G.Bytes(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(dev_supsyn_data, _connectivity._supsyn._data, _connectivity._supsyn.Bytes(),
                            cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(dev_actsyn_data, _connectivity._actsyn._data, _connectivity._actsyn.Bytes(),
                            cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Restore references
    _connectivity._actcount = d_actcount;
    _connectivity._supcount = d_supcount;
    _connectivity._NSS = d_NSS;
    _connectivity._G._data = dev_G_data;
    _connectivity._supsyn._data = dev_supsyn_data;
    _connectivity._actsyn._data = dev_actsyn_data;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // End timers
    float elapsedKernelTime;
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedKernelTime, start, stop));

    LOG("Synaptic Decay Kernel ~fin %3.1f ms", elapsedKernelTime)
    return elapsedKernelTime;
}


float CUSynfire::MembranePotentialLauncher() {
    LOG("Preparing Membrane Potential Kernel")
    int numThreads = 256;
    int numBlocks = MIN( 32, (network_size + numThreads - 1) / numThreads );
    if(numBlocks <= 0) numBlocks = 1;
    if (network_size % numThreads == 0) ++numBlocks;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Start timers
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Prepare data
    int ran_calls = network_size * 4;
    for(int i = 0; i < ran_calls; ++i) _ranCache[i] = ran1(&seed);
    HANDLE_ERROR(cudaMemcpy(_dranCache, _ranCache, sizeof(float) * network_size * 4, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(_dnetwork, _network, sizeof(Neuron) * network_size, cudaMemcpyHostToDevice));

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Launch kernel
    LOG("Launching Membrane Potential Kernel")
    cudaDeviceSynchronize();
    MembranePotentialKernel<<<numBlocks, numThreads>>>(DT, _dnetwork, network_size, _dspikeFlags, _dranCache);
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    //~ Load results.
    HANDLE_ERROR(cudaMemcpy(_network, _dnetwork, sizeof(Neuron) * network_size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(_spikeFlags, _dspikeFlags, sizeof(bool) * network_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < network_size; ++i) {
        if (_spikeFlags[i] == true) {
            _whospiked.push_back(i);
        }
    }

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // End timers
    float elapsedKernelTime;
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedKernelTime, start, stop));
    LOG("Membrane Potential Kernel ~fin  %3.1f ms", elapsedKernelTime)
    return elapsedKernelTime;
}


Synapses *CUSynfire::CreateDeviceSynapses( Synapses *syn ) {
    Synapses *dsyn;

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Create and copy class object.
    HANDLE_ERROR(cudaMalloc(&dsyn, sizeof(Synapses)));
    HANDLE_ERROR(cudaMemcpy(dsyn, syn, sizeof(Synapses), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Allocate device actcount array.
    int *d_actcount;
    HANDLE_ERROR(cudaMalloc((void **) &d_actcount, sizeof(int) * network_size));
    HANDLE_ERROR(cudaMemcpy(&(dsyn->_actcount), &d_actcount, sizeof(int *), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_actcount, syn->_actcount, sizeof(int) * network_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Allocate device supcount array.
    int *d_supcount;
    HANDLE_ERROR(cudaMalloc((void **) &d_supcount, sizeof(int) * network_size));
    HANDLE_ERROR(cudaMemcpy(&(dsyn->_supcount), &d_supcount, sizeof(int *), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_supcount, syn->_supcount, sizeof(int) * network_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Allocate device NSS array.
    int *d_NSS;
    HANDLE_ERROR(cudaMalloc((void **) &d_NSS, sizeof(int) * network_size));
    HANDLE_ERROR(cudaMemcpy(&(dsyn->_NSS), &d_NSS, sizeof(int *), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_NSS, syn->_NSS, sizeof(int) * network_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Allocate device G (synapse strength) array.
    double *dev_G_data;
    HANDLE_ERROR(cudaMalloc((void **) &dev_G_data, _connectivity._G.Bytes()));
    HANDLE_ERROR(cudaMemcpy(&(dsyn->_G._data), &dev_G_data, sizeof(double *), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_G_data, _connectivity._G._data, _connectivity._G.Bytes(), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Allocate device supsyn (super synapses) array.
    bool *dev_supsyn_data;
    HANDLE_ERROR(cudaMalloc((void **) &dev_supsyn_data, _connectivity._supsyn.Bytes()));
    HANDLE_ERROR(cudaMemcpy(&(dsyn->_supsyn._data), &dev_supsyn_data, sizeof(double *), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_supsyn_data, _connectivity._supsyn._data, _connectivity._supsyn.Bytes(),
                            cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    //"""""""""""""""""""""""""""""""""""""""""""""""""
    // Allocate device actsyn (active synapses) array.
    bool *dev_actsyn_data;
    HANDLE_ERROR(cudaMalloc((void **) &dev_actsyn_data, _connectivity._actsyn.Bytes()));
    HANDLE_ERROR(cudaMemcpy(&(dsyn->_actsyn._data), &dev_actsyn_data, sizeof(double *), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_actsyn_data, _connectivity._actsyn._data, _connectivity._actsyn.Bytes(),
                            cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    LOG("Copied data to device.")
    return dsyn;
}

__global__
void SynapticDecayKernel( Synapses *dconnectivity, int syn_size ) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < syn_size) {
        int pre = i / syn_size;
        double syndecay = dconnectivity->GetSynDecay();
        for (int j = 0; j < syn_size; ++j) {
            double syn_str_dec = dconnectivity->GetSynapticStrength(pre, j) * syndecay;
            dconnectivity->CheckThreshold(syn_str_dec, pre, j, 'a', 'd');
            dconnectivity->CheckThreshold(syn_str_dec, pre, j, 's', 'd');
            dconnectivity->SetSynapticStrength(pre, j, syn_str_dec);
        }
    }
}

__global__
void MembranePotentialKernel( float dt, Neuron *net, int net_size, bool *whospiked, float *dranCache ) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < net_size) {
        whospiked[tid] = net[tid].Update(dt, dranCache[tid], dranCache[tid + 1], dranCache[tid + 2], dranCache[tid + 3]);
    }

}

#pragma clang diagnostic pop
