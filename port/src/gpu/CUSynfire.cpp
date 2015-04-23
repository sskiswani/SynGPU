#include "CUSynfire.h"
#include "cuda_utils.h"
//#include <cuda_runtime.h>
#include <vector>
#include "random.h"
#include "microtime.h"
#include "neuron.h"
#include "synapses.h"

void CUSynfire::Initialize() {
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
	
	// Allocate and initialize host copy of the network
    _network = new Neuron[network_size];
    for (int i = 0; i < network_size; ++i) {
        _network[i] = Neuron(i, exfreq, infreq, examp, inamp, global_i);
    }
	// Allocate and initialize host copy of Synapses objects	
	_connectivity = Synapses(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn);
    _inhibition_strength = Synapses(frac, 0.0, act, sup, cap, syndec, conn_type, network_size, tempNSS, window, eq_syn);

    //~ Allocate and initialize device copy of the network.
    HANDLE_ERROR(cudaMalloc((void **) &_dnetwork, sizeof(Neuron) * network_size));
    HANDLE_ERROR(cudaMemcpy(_dnetwork, _network, sizeof(Neuron) * network_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void **) &_dconnectivity, sizeof(Synapses));
	HANDLE_ERROR(cudaMemcpy(_dconnectivity, &_connectivity, sizeof(Synapses), cudaMemcpyHostToDevice));
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
			SynapticDecayWrapper();
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

void CUSynfire::RunTrial( double *tT, double *tTS, double *tMPL, double *tSL, double *tTSa ) {
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
//            int *send_to;//pointer to array containing post neurons
//            int send_count;//number of post neurons receiving spike
//            cout<<v_whospike[j]<<" spikes!!! at "<<t<<endl;
//            _network[v_whospike[j]].recspike(t); //record time of spike
            sc++; //keep track of total number of spikes this trial

            // TODO: L1348: Emit spikes
        }

        tSL[1] = microtime();
        tSL[2] += (tSL[1] - tSL[0]);
        // cout << "SL current: " << tSL[1]-tSL[0] << " SL total:" << tSL[2] << endl;

        // L1372: Inhibition.
        inh[dsteps - 1] = whocount;
        for (int z = 0; z < whocount; ++z) {
            for (int j = 0; j < network_size; ++j) {
                // network[j]->excite_inhibit(inhibition_strength->getsynstr(v_whospike[z],j),'i');
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

__global__
void CUSynfire::SynapticDecayKernel(int syn_size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < syn_size) {
		int pre = i / syn_size;
		double syndecay = _dconnectivity->GetSynDecay();
		for (int j = 0; j < syn_size; ++j) {
			int syn_str = _dconnectivity->_G[j]*syndecay;
			connectivity->CheckThreshold(syn_str, pre, j, 'a', 'd');
			connectivity->CheckThreshold(syn_str, pre, j, 's', 'd');
		}
	}	
}

void CUSynfire::SynapticDecayWrapper() {
	// Intermediate host pointers
	bool *actsyn, *supsyn;
	int *actcount, *supcount;
	double *G;
	double *syndecay, *actthres, *supthres;

	int syn_size = network_size*network_size;

	int numThreads = 256;
	int numBlocks = syn_size / numThreads;
	if (syn_size % numThreads == 0) ++numBlocks;

	// start timers
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// Transfer Host To Device
	HANDLE_ERROR(cudaMemcpy(&(_dconnectivity->_actcount), &actcount, sizeof(int*), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(actcount, _connectivity->_actcount, sizeof(int)*network_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(&(_dconnectivity->_supcount), &supcount, sizeof(int*), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(supcount, _connectivity->_supcount, sizeof(int)*network_size, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(&(_dconnectivity->_actsyn), &actsyn, sizeof(bool*), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(actsyn, _connectivity->_actsyn, sizeof(bool)*syn_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(&(_dconnectivity->_supsyn), &supsyn, sizeof(bool*), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(supsyn, _connectivity->_supsyn, sizeof(bool)*syn_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(&(_dconnectivity->_G), &G, sizeof(double*), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(G, _connectivity->_G, sizeof(double)*syn_size, cudaMemcpyHostToDevice));

	SynapticDecayKernel<<<numBlocks, numThreads>>>(syn_size);

	// Transfer Device To Host
	HANDLE_ERROR(cudaMemcpy(&actcount, &(_dconnectivity->_actcount), sizeof(int*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(_connectivity->_actcount, actcount, sizeof(int)*network_size, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&supcount, &(_dconnectivity->_supcount), sizeof(int*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(_connectivity->_supcount, supcount, sizeof(int)*network_size, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaMemcpy(&actsyn, &(_dconnectivity->_actsyn), sizeof(bool*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(_connectivity->_actsyn, actsyn, sizeof(bool)*syn_size, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&supsyn, &(_dconnectivity->_supsyn), sizeof(bool*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(_connectivity->_supsyn, supsyn, sizeof(bool)*syn_size, cudaMemcpyDeviceToHost));

	cudaFree(actsyn); 		cudaFree(supsyn); 
	cudaFree(actcount); 	cudaFree(supcount);
	cudaFree(G);
	
	// End timers
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, stop));
   	printf( "Time taken:  %3.1f ms\n", elapsedTime);
}

//-----------------------------Synapses Host/Device Implementations-----------------------------//

__host__ __device__
void Synapses::Deactivate( char syn_type, int pre, int post ) {
    // 'a' for active, 's' for super
    if (syn_type == 'a') {
		--(_actcount[pre]);
		_actsyn[pre * _size + post] = false;	
    } else if (syn_type == 's') {
		--(_supcount[pre]);
		_supsyn[pre * _size + post] = false;
    } else {
        // Invalid argument
        return;
    }
}

__host__ __device__
void Synapses::CheckThreshold( double syn_str, int pre, int post, char syn_type, char pd_type ) {
    //p=='a'/p=='s', check if G crossed active/super threshold
    //q == 'p' for potentiation, q == 'd' for depression
    double thres;
    if (syn_type == 'a') thres = _actthres;
    else if (pd_type == 's') thres = _supthres;
    else return;// ERROR

    if (pd_type == 'p') {
        if (_G[pre * _size + post] < thres && syn_str >= thres) {
            Activate(syn_type, pre, post);
        }
    } else if (pd_type == 'd') {
        if (_G[pre * _size + post] >= thres && syn_str < thres) {
            Deactivate(syn_type, pre, post);
        }
    } else {
        // ERROR
        return;
    }
}
