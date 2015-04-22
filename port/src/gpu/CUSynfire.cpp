#include "CUSynfire.h"
#include "cuda_utils.h"
//#include <cuda_runtime.h>
#include <vector>
#include "random.h"
#include "microtime.h"
#include "neuron.h"

void CUSynfire::Initialize() {

    _network = new Neuron[network_size];
    for (int i = 0; i < network_size; ++i) {
        _network[i] = Neuron(i, exfreq, infreq, examp, inamp, global_i);
    }

    //~ Allocate and initialize device copy of the network.
    HANDLE_ERROR(cudaMalloc((void **) &_dnetwork, sizeof(Neuron) * network_size));
    HANDLE_ERROR(cudaMemcpy(_dnetwork, _network, sizeof(Neuron) * network_size, cudaMemcpyHostToDevice));
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
