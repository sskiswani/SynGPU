#include "neuron.h"
#include "random.h"

// Helpers
int Neuron::_LABEL_COUNTER = 0;

// Spontaneous activity defaults.
const double Neuron::DEFAULT_EXFREQ = 40;
const double Neuron::DEFAULT_INFREQ = 200;
const double Neuron::DEFAULT_EXAMP = 1.3;
const double Neuron::DEFAULT_INAMP = 0.1;
const double Neuron::DEFAULT_GLOBAL_I = 0.3;
const double Neuron::DEFAULT_INHDECAY = 0.0;
const double Neuron::DEFAULT_LEAK = -85.0;

// Integrate and Fire model parameters. (in ms)
const double Neuron::DECAY_INHIBITORY = 3.0;    // Inhibitory
const double Neuron::DECAY_EXCITORY = 5.0;      // Excitory
const double Neuron::DECAY_MEMBRANE = 20.0;     // Membrane
const double Neuron::SPIKE_THRESHOLD = -50.0;        // Spike threshold (mV)
const double Neuron::RESET = -80.0;                   // Membrane reset potential (mV)
const double Neuron::INHIBITORY_REVERSAL = -75.0;   // potential (mV)
const double Neuron::REFCTORY_TIME = 25.0;           // ms
const double Neuron::LATENCY_TIME = 2.0;             // ms


Neuron::Neuron() {
    Initialize(++_LABEL_COUNTER, DEFAULT_EXFREQ, DEFAULT_INFREQ, DEFAULT_EXAMP, DEFAULT_INAMP, DEFAULT_GLOBAL_I,
               DEFAULT_LEAK);
}

Neuron::Neuron( int label,
                double exc_freq,
                double inh_freq,
                double exc_amp,
                double inh_amp,
                double global_inhibition ) {
    Initialize(label, exc_freq, inh_freq, exc_amp, inh_amp, global_inhibition, DEFAULT_LEAK);
//    SetSpinFrequency(exc_freq, inh_freq);
//    _spamp_ex = exc_amp;
//    _spamp_in = inh_amp;
//
//    _global_in = global_inhibition;
//
//    _label = label;
//    _cLatent = 0;
//    _cRef = 0;
//    _cSpike = 0;
//
//    Reset();
//    _LEAKREV = Neuron::DEFAULT_LEAK;
}

bool Neuron::Update( float dt ) {
    bool spike = false;

    //spontaneous excitation and inhibition
    if (ran1(&seed) < dt * _spfreq_ex) ExciteInhibit(_spamp_ex * ran1(&seed), 'e');
    if (ran1(&seed) < dt * _spfreq_in) ExciteInhibit(_spamp_in * ran1(&seed), 'i');


    // if neuron isn't in latent period before spike
    if (_cLatent < 1 && _cRef < 1) {
        neur_dyn(dt, false);

        // go into latency before spike if potential > threshold & not refractory
        if (_volts >= Neuron::SPIKE_THRESHOLD) {
            _volts = 0;
            _cLatent = (int) (Neuron::LATENCY_TIME / dt);
        }
    } else {
        // update refractory period counter.
        if (_cRef >= 1) --_cRef;

        if (_cLatent >= 1) {
            --_cLatent; // update counter.

            if (_volts == 0) _volts = Neuron::RESET;

            // Spike if the latency timer ends on this step
            if (_cLatent < 1) {
                _cLatent = 0;
                _cRef = (int) (Neuron::REFCTORY_TIME / dt);
                spike = true;
            }
        }

        neur_dyn(dt, true);
    }

    return spike;
}

void Neuron::neur_dyn( double dt, bool no_volt ) {
    //update membrane potential,conductances with 4th order RK
    double c1[3], c2[3], c3[3], c4[3];
    c1[0] = -dt * _gexc / Neuron::DECAY_EXCITORY;
    c1[1] = -dt * _ginh / Neuron::DECAY_INHIBITORY;
    if (no_volt == false) {
        c1[2] = (dt / Neuron::DECAY_MEMBRANE) *
                ((_LEAKREV - _volts) - _gexc * _volts + _ginh * (Neuron::INHIBITORY_REVERSAL - _volts));
    }
    else {
        c1[2] = 0;
    }

    c2[0] = -dt * (_gexc + c1[0] / 2.0) / Neuron::DECAY_EXCITORY;
    c2[1] = -dt * (_ginh + c1[1] / 2.0) / Neuron::DECAY_INHIBITORY;
    if (no_volt == false) {
        c2[2] = (dt / Neuron::DECAY_MEMBRANE) *
                (_LEAKREV - (_volts + (c1[2] / 2.0)) - (_gexc + c1[0] / 2.0) * (_volts + (c1[2] / 2.0)) +
                 (_ginh + c1[1] / 2.0) * (Neuron::INHIBITORY_REVERSAL - (_volts + (c1[2] / 2.0))));
    }
    else {
        c2[2] = 0;
    }

    c3[0] = -dt * (_gexc + c2[0] / 2.0) / Neuron::DECAY_EXCITORY;
    c3[1] = -dt * (_ginh + c2[1] / 2.0) / Neuron::DECAY_INHIBITORY;
    if (no_volt == false) {
        c3[2] = (dt / Neuron::DECAY_MEMBRANE) *
                (_LEAKREV - (_volts + (c2[2] / 2.0)) - (_gexc + c2[0] / 2.0) * (_volts + (c2[2] / 2.0)) +
                 (_ginh + c2[1] / 2.0) * (Neuron::INHIBITORY_REVERSAL - (_volts + (c2[2] / 2.0))));
    }
    else {
        c3[2] = 0;
    }

    c4[0] = -dt * (_gexc + c3[0]) / Neuron::DECAY_EXCITORY;
    c4[1] = -dt * (_ginh + c3[1]) / Neuron::DECAY_INHIBITORY;
    if (no_volt == false) {
        c4[2] = (dt / Neuron::DECAY_MEMBRANE) * (_LEAKREV - (_volts + c3[2]) - (_gexc + c3[0]) * (_volts + (c3[2])) +
                                                 (_ginh + c3[1]) * (Neuron::INHIBITORY_REVERSAL - (_volts + c3[2])));
    }
    else {
        c4[2] = 0;
    }

    _volts += (c1[2] + 2 * c2[2] + 2 * c3[2] + c4[2]) / 6.0;
    _gexc += (c1[0] + 2 * c2[0] + 2 * c3[0] + c4[0]) / 6.0;
    _ginh += (c1[1] + 2 * c2[1] + 2 * c3[1] + c4[1]) / 6.0;
}

void Neuron::ExciteInhibit( double amp, char p ) {
    if (p == 'e') _gexc += amp;
    else if (p == 'i') _ginh += amp;
}

/**
 * Reset the Neuron to a random state.
 */
void Neuron::Reset() {
    static double gexc_avg = 0.5 * Neuron::DECAY_EXCITORY * _spamp_ex * _spfreq_ex;
    static double ginh_avg = 0.5 * Neuron::DECAY_INHIBITORY * _spamp_in * _spfreq_in;

    _volts = ran1(&seed) * (-55 + 80) - 80;

    // Select a conductance
    _gexc = 2 * gexc_avg * ran1(&seed);
    _ginh = 2 * ginh_avg * ran1(&seed);
}


void Neuron::RecordSpike( double t ) {
    // Record this spike in _spkhist.
//    _spkhist[_cSpike] = t;
    ++_cSpike;
}

void Neuron::ResetSpike() {
    // TODO: How to store spike history?
    if (_cSpike != 0) {
        _cSpike = 0;
        delete[] _spkhist;
    }
}

/**
 * Initialize Neuron data.
 */
void Neuron::Initialize( int label,
                         double exc_freq,
                         double inh_freq,
                         double exc_amp,
                         double inh_amp,
                         double global_inhibition,
                         double leak ) {
    SetSpinFrequency(exc_freq, inh_freq);
    _spamp_ex = exc_amp;
    _spamp_in = inh_amp;

    _global_in = global_inhibition;

    _label = label;
    _cLatent = 0;
    _cRef = 0;
    _cSpike = 0;

    Reset();
    _LEAKREV = leak;
}
