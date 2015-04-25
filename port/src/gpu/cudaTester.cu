#include "helpers.h"
#include "utility.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
__global__ void print_kernel( TArray2<int> *dtester ) {
    if (blockIdx.x == threadIdx.x)
        ((*dtester)(threadIdx.x, blockIdx.x)) = 1337;
}

#pragma clang diagnostic pop

int main( int argc, char *argv[] ) {
    //~ Prepare device.
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
    struct Timer timer;


    TArray2<int> tester(10, 10);
    for (int y = 0, i = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x, ++i) {
            tester(x, y) = i;
        }
    }

    std::cerr << "Tester:\n" << tester;

    std::cerr << "Copying to device..." << std::endl;
    TArray2<int> *dtester = tester.CopyToDevice();
    cudaDeviceSynchronize();

    std::cerr << "Running kernel now..." << std::endl;

    print_kernel <<<10, 10>>> ((TArray2<int> *) (dtester));
    cudaDeviceSynchronize();

    std::cerr << "Copying to host..." << std::endl;

    TArray2<int> helper;
    HANDLE_ERROR(cudaMemcpy(&helpet, dtester, sizeof(TArray2<int>), cudaMemcpyDeviceToHost));

    int* t_data = tester._data;
    HANDLE_ERROR(cudaMemcpy(tester._data, helper.data, tester.Bytes(), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();


    std::cerr << "Tester:\n" << tester;

}
