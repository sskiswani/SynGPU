//#include <stdlib.h>
//#include <math.h>
//#include <time.h>
//#include <fstream>
//#include <sstream>
//#include <string>
//#include "microtime.h"

#include <iostream>

#include "random.h"
#include "utility.h"
#include "synfire.h"
#include "neuron.h"


// TODO: Needs to be properly located
const int NETWORK_SIZE = 200;

// TODO: Spontaneous activity defaults

//Training defaults
int ntrain = 10, ntrg = 1; //number of training neurons
bool man_tt = false; //manual training time bool (false if training occurs at t=0)
double *train_times;
double training_f = 1.5; //training spike frequency (ms)^(-1)
double training_amp = .7; //training spike strength
double training_t = 8.0; //training duration in ms

//Stats
bool stats_on = true;
int sc = 0; //total spike counter
double av = 0;
double rock, roll;

int main( int argc, char *argv[] ) {
    struct Timer timer;

    // TODO: Command line arguments

    // TODO: Allocate Synpses & Neurons
    timer.Start();
    Synfire syn(200);
    timer.Stop();
    std::cout << "Initialization time: " << timer.Duration() << " ms." << std::endl;

    // TODO: Seed generators

    // TODO: Initialize Synapse Object

    // TODO: Prepare trials

    syn.Run();
}
