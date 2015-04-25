#include "cu_synfire.cuh"

int main(int argc, char *argv[]) {
    std::cout << "Welcome to SynfireGrowth powered by NVIDIA - CUDA.\n" << std::endl;
    SynfireParameters params(argc, argv);

    //~ Prepare device.
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice) );
    Timer timer;

    // TODO: Command line arguments

    timer.Start();
    CUSynfire synfire = CUSynfire::CreateCUSynfire();
    timer.Stop();

    std::cout << "Initialization time: " << US_TO_MS(timer.Duration()) << " ms." << std::endl;

    synfire.Run();
}
