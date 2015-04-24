#include <iostream>

#include "utility.h"
#include "synfire.h"

int main( int argc, char *argv[] ) {
    struct Timer timer;

    // TODO: Command line arguments

    timer.Start();
    Synfire syn = Synfire::CreateSynfire();
    timer.Stop();
    std::cout << "Initialization time: " << US_TO_MS(timer.Duration()) << " ms." << std::endl;

    syn.Run();
}
