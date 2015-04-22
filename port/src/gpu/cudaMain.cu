//#include <stdlib.h>
//#include <math.h>
//#include <time.h>
//#include <string>
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include "microtime.h"

#include "cuda_utils.h"


int main(int argc, char *argv[]) {
    //~ Prepare device.
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice) );

    // TODO: Command line arguments

    // TODO: Allocate Synapses & Neurons

    // TODO: Seed generators

    // TODO: Initialize Synapse Object
}
