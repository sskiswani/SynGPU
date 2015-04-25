//#include <stdlib.h>
//#include <math.h>
//#include <time.h>
//#include <string>
//#include <iostream>
//#include <fstream>
//#include <sstream>

#include <stdio.h>
#include <iostream>
#include "helpers.h"
#include "cuda_utils.h"
#include "utility.h"

__global__ void print_kernel(TArray2<int> *dtester) {
    if(blockIdx.x < 10 && threadIdx.x < 10) {
        printf("Hello from block{%d} -> thread{%d} which got %i\n",
               blockIdx.x, threadIdx.x,
               (*dtester)(threadIdx.x, blockIdx.x));
    }
}

int main(int argc, char *argv[]) {
    TArray2<int> tester(10, 10);
    for(int y = 0, i = 0; y < 10; ++y) {
        for(int x = 0; x < 10; ++x, ++i) {
            tester(x, y) = i;
            printf("CPU test: (%i,%i)=%i\n", x, y, tester(x,y));
        }
    }

    std::cerr << "Copying to device..." << std::endl;

    void *dtester = tester.CopyToDevice();

    std::cerr << "Running kernel now..." << std::endl;

    //~ Prepare device.
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice) );
    struct Timer timer;

    print_kernel<<<10, 10>>>((TArray2<int> *)(dtester));
    cudaDeviceSynchronize();
}
