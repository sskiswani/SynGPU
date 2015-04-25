#include <iostream>

#include "utility.h"
#include "synfire.h"

int main( int argc, char *argv[] ) {
    SynfireParameters params(argc, argv);
    Timer timer;

    timer.Start();
    Synfire syn(params);
    timer.Stop();
    std::cout << "Initialization time: " << US_TO_MS(timer.Duration()) << " ms." << std::endl;

    syn.Run();
}
