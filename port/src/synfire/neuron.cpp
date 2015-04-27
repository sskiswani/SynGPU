#include "neuron.h"
#include "random.h"

// Helpers
int Neuron::_LABEL_COUNTER = 0;


Neuron::Neuron() {
    Initialize(++_LABEL_COUNTER,
               NEURON_DEFAULT_EXFREQ,
               NEURON_DEFAULT_INFREQ,
               NEURON_DEFAULT_EXAMP,
               NEURON_DEFAULT_INAMP,
               NEURON_DEFAULT_GLOBAL_I,
               NEURON_DEFAULT_LEAK);
}

Neuron::Neuron( int label,
                double exc_freq,
                double inh_freq,
                double exc_amp,
                double inh_amp,
                double global_inhibition ) {
    Initialize(label, exc_freq, inh_freq, exc_amp, inh_amp, global_inhibition, NEURON_DEFAULT_LEAK);
}

Neuron::~Neuron() {

}

bool Neuron::Update( float dt ) {
    return Update(dt, ran1(&seed), ran1(&seed), ran1(&seed), ran1(&seed));
}

CUDA_CALLABLE bool Neuron::Update( float dt, float r1, float r2, float r3, float r4 ) {
    bool spike = false;

    //spontaneous excitation and inhibition
    if (r1 < dt * _spfreq_ex) ExciteInhibit(_spamp_ex * r2, 'e');
    if (r3 < dt * _spfreq_in) ExciteInhibit(_spamp_in * r4, 'i');


    // if neuron isn't in latent period before spike
    if (_cLatent < 1 && _cRef < 1) {
        neur_dyn(dt, false);

        // go into latency before spike if potential > threshold & not refractory
        if (_volts >= NEURON_SPIKE_THRESHOLD) {
            _volts = 0;
            _cLatent = (int) (NEURON_LATENCY_TIME / dt);
        }
    } else {
        // update refractory period counter.
        if (_cRef >= 1) --_cRef;

        if (_cLatent >= 1) {
            --_cLatent; // update counter.

            if (_volts == 0) _volts = NEURON_RESET;

            // Spike if the latency timer ends on this step
            if (_cLatent < 1) {
                _cLatent = 0;
                _cRef = (int) (NEURON_REFCTORY_TIME / dt);
                spike = true;
            }
        }

        neur_dyn(dt, true);
    }

    return spike;
}

CUDA_CALLABLE void Neuron::neur_dyn( double dt, bool no_volt ) {
    const double DECAY_EXCITORY = NEURON_DECAY_EXCITORY;
    const double DECAY_MEMBRANE = NEURON_DECAY_MEMBRANE;
    const double DECAY_INHIBITORY = NEURON_DECAY_INHIBITORY;
    const double INHIBITORY_REVERSAL = NEURON_INHIBITORY_REVERSAL;


	double temp0, temp1, temp2;
	double tVolts = 0, tGexh = 0, tGinh = 0;

	temp2 = (dt/DECAY_MEMBRANE)*((_LEAKREV-_volts)-_gexc*_volts+_ginh*(INHIBITORY_REVERSAL-_volts));
	temp2 = temp2 - (int)no_volt * temp2;
	temp1 =-dt*_ginh/DECAY_INHIBITORY;
	temp0 =-dt*_gexc/DECAY_EXCITORY;

	tVolts += temp2;
	tGexh += temp0;
	tGinh += temp1;

	temp2 =  (dt/DECAY_MEMBRANE)*(_LEAKREV-(_volts+(temp2/2.0))-(_gexc+temp0/2.0)*(_volts+(temp2/2.0))+(_ginh+temp1/2.0)*(INHIBITORY_REVERSAL-(_volts+(temp2/2.0))));
	temp2 = temp2 -(int)no_volt * temp2;
	temp1 = -dt*(_ginh+temp1/2.0)/DECAY_INHIBITORY;
	temp0 = -dt*(_gexc+temp0/2.0)/DECAY_EXCITORY;

	tVolts += (2*temp2);
	tGexh += (2*temp0);
	tGinh += (2*temp1);

	temp2 =  (dt/DECAY_MEMBRANE)*(_LEAKREV-(_volts+(temp2/2.0))-(_gexc+temp0/2.0)*(_volts+(temp2/2.0))+(_ginh+temp1/2.0)*(INHIBITORY_REVERSAL-(_volts+(temp2/2.0))));
	temp2 = temp2 -(int)no_volt * temp2;
	temp1 = -dt*(_ginh+temp1/2.0)/DECAY_INHIBITORY;
	temp0 = -dt*(_gexc+temp0/2.0)/DECAY_EXCITORY;

	tVolts += (2*temp2);
	tGexh += (2*temp0);
	tGinh += (2*temp1);

	temp2 =  (dt/DECAY_MEMBRANE)*(_LEAKREV-(_volts+(temp2/2.0))-(_gexc+temp0/2.0)*(_volts+(temp2/2.0))+(_ginh+temp1/2.0)*(INHIBITORY_REVERSAL-(_volts+(temp2/2.0))));
	temp2 = temp2 -(int)no_volt * temp2;
	temp1 = -dt*(_ginh+temp1/2.0)/DECAY_INHIBITORY;
	temp0 = -dt*(_gexc+temp0/2.0)/DECAY_EXCITORY;

	_volts += (tVolts+temp2)/6.0;
	_gexc += (tGexh+temp0)/6.0;
	_ginh += (tGinh+temp1)/6.0;
}


CUDA_CALLABLE void Neuron::ExciteInhibit( double amp, char p ) {
    if (p == 'e') _gexc += amp;
    else if (p == 'i') _ginh += amp;
}

/**
 * Reset the Neuron to a random state.
 */
void Neuron::Reset() {
    static double gexc_avg = 0.5 * NEURON_DECAY_EXCITORY * _spamp_ex * _spfreq_ex;
    static double ginh_avg = 0.5 * NEURON_DECAY_INHIBITORY * _spamp_in * _spfreq_in;

    _volts = ran1(&seed) * (-55 + 80) - 80;

    // Select a conductance
    _gexc = 2 * gexc_avg * ran1(&seed);
    _ginh = 2 * ginh_avg * ran1(&seed);
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

    Reset();
    _LEAKREV = leak;
}
