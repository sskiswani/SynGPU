#include <iostream>

#include "utility.h"
#include "synfire.h"

int main( int argc, char *argv[] ) {
    struct Timer timer;

    // TODO: Command line arguments

    // TODO: Allocate Synpses & Neurons
    timer.Start();
    Synfire syn = Synfire::CreateSynfire();
    timer.Stop();
    std::cout << "Initialization time: " << timer.Duration() << " ms." << std::endl;

    // TODO: Seed generators

    // TODO: Initialize Synapse Object

    // TODO: Prepare trials

    syn.Run();
}
