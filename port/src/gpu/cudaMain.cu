//#include <stdlib.h>
//#include <math.h>
//#include <time.h>
//#include <string>
//#include <iostream>
//#include <fstream>
//#include <sstream>

#include <iostream>
#include "utility.h"
#include "cu_synfire.cuh"
#include "cuda_utils.h"


int main(int argc, char *argv[]) {
    std::cout << "Welcome to SynfireGrowth powered by NVIDIA - CUDA.\n" << std::endl;

    //~ Prepare device.
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice) );
    struct Timer timer;

    // TODO: Command line arguments

    timer.Start();
    CUSynfire synfire = CUSynfire::CreateCUSynfire();
    timer.Stop();
    std::cout << "Initialization time: " << US_TO_MS(timer.Duration()) << " ms." << std::endl;

    synfire.Run();
}
