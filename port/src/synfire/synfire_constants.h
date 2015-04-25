#ifndef PORT_SYNFIRE_CONSTANTS_H
#define PORT_SYNFIRE_CONSTANTS_H

// Spontaneous activity defaults.
#define NEURON_DEFAULT_EXFREQ    40
#define NEURON_DEFAULT_INFREQ    200
#define NEURON_DEFAULT_EXAMP     1.3
#define NEURON_DEFAULT_INAMP     0.1
#define NEURON_DEFAULT_GLOBAL_I  0.3
#define NEURON_DEFAULT_INHDECAY  0.0
#define NEURON_DEFAULT_LEAK      -85.0

// Integrate and Fire model parameters. (in ms)
#define NEURON_DECAY_INHIBITORY     3.0     // Inhibitory
#define NEURON_DECAY_EXCITORY       5.0     // Excitory
#define NEURON_DECAY_MEMBRANE       20.0    // Membrane
#define NEURON_SPIKE_THRESHOLD      -50.0   // Spike threshold (mV)
#define NEURON_RESET                -80.0   // Membrane reset potential (mV)
#define NEURON_INHIBITORY_REVERSAL  -75.0   // potential (mV)
#define NEURON_REFCTORY_TIME        25.0    // ms
#define NEURON_LATENCY_TIME         2.0     // ms

#endif //PORT_SYNFIRE_CONSTANTS_H
