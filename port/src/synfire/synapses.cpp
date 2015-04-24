#include <math.h>
#include "random.h"
#include "synapses.h"
#include "neuron.h"

// Synaptic Plasticity parameters
const double Synapses::AMPLITUDE_LTP = .01;            // Long-term Plasticity amplitude
const double Synapses::AMPLITUDE_LTD = .0105;        // Long-term Depression amplitude

const double Synapses::INV_DECAY_LTP = .05;            // Long-term Plasticity Inverse Decay times(ms)
const double Synapses::INV_DECAY_LTD = .05;            // Long-term Depression Inverse Decay times(ms)

const double Synapses::POTENTIATION_FUNT = 5.0;        // Potentiation
const double Synapses::DEPRESSION_FUNT = 5.25;        // Depression


Synapses::Synapses( double fract_act,
                    double glob,
                    double act_thres,
                    double sup_thres,
                    double syn_max,
                    double syn_decay,
                    int form_opt,
                    int network_size,
                    int tempNSS,
                    double window,
                    double eq_syn )
        : _size(network_size),
          _G(_size, _size),
          _actsyn(_size, _size),
          _supsyn(_size, _size) {
    _window = window;
    _GLTP = eq_syn;

    _actthres = act_thres;
    _supthres = sup_thres;

    _synmax = syn_max;
    _syndec = syn_decay;

    // Allocate data
    _actcount = new int[_size];
    _supcount = new int[_size];

    _NSS = new int[_size];

    for (int i = 0; i < _size; ++i) {
        _NSS[i] = tempNSS;
        _actcount[i] = 0;
        _supcount[i] = 0;
    }

    if (form_opt == 1) { // Randomly activate fract_act of all synapses
        double value;

        for (int pre = 0; pre < _size; ++pre) {
            for (int post = 0; post < _size; ++post) {
                if (pre == post) continue;

                if (ran1(&seed) <= fract_act) {
                    value = _actthres + (_synmax - _actthres) * ran1(&seed);
                    Activate('a', pre, post);
                } else {
                    value = _actthres * ran1(&seed);
                }

                _G(post, pre) = (value > _synmax) ? _synmax : value;
            }
        }
    } else { // Default initialization.
        for (int i = 0; i < _size; ++i) {
            for (int j = 0; j < _size; ++j) {
                _G(i, j) = glob;
            }
        }
    }
}

/**
 *	Main problem is allocating space on device (how to transfer correct amount from device2host)
 *	To resolve issue: have kernel identify total number of activated synapse per spiked neuron
 *	then on host or kernel allocate space
 *
 *	Create a special activate/deactivate kernel function specifically for synaptic_decay and synaptic_plasticity
 *
 *  To avoid thread divergence for the first two switch statements and last if statement
 *  have _act_syn(in constructor or wherever it happens) be allocated some amount of memory(regardless if empty)
 *  so we can get rid of the if(*l != 0) statment
 *
 *  Activate a synapse between two neurons.
 *
 *  @param p        'a' for active, 's' for super
 *  @param pre      pre-neuron label
 *  @param post     post-neuron label
 */
void Synapses::Activate( char p, int pre, int post ) {
    if (p == 'a') {
        ++(_actcount[pre]);
        _actsyn(post, pre) = true;
    } else if (p == 's') {
        ++(_supcount[pre]);
        _supsyn(post, pre) = true;
    }
}

/**
 *  Deactivate a synapse between two neurons.
 *
 *  @param p        'a' for active, 's' for super
 *  @param pre      pre-neuron label
 *  @param post     post-neuron label
 */
void Synapses::Deactivate( char p, int pre, int post ) {
    if (p == 'a') {
        --(_actcount[pre]);
        _actsyn(post, pre) = false;
    } else if (p == 's') {
        --(_supcount[pre]);
        _supsyn(post, pre) = false;
    }
}

/**
 * Synaptic_Plasticity.
 *
 * @param spiker        Label of the neuron that spiked.
 * @param t             Elapsed time of the current trial.
 * @param spk_times     A log of the times the spiking neuron has previous spiked.
 * @param spk_count     Size of the spk_times array.
 */
void Synapses::Synaptic_Plasticity( int spiker, double t, double *spk_times, int spk_count ) {
    double tempPot, tempDep, GPot, GDep;

    for (int k = 0; k < _size; ++k) {
        if (k == spiker) continue;

        // TODO: Are these method calls correct? Figure out correct axes.
        GPot = _G(spiker, k);
        GDep = _G(k, spiker);

        if (spk_count != 0 && spk_times[spk_count - 1] + _window >= t) {
            tempPot = GPot + AMPLITUDE_LTP * _GLTP * PotentiationFunc(t, spk_count, spk_times, 'p'); //potentiation
            tempDep = GDep * (1 - AMPLITUDE_LTP * PotentiationFunc(t, spk_count, spk_times, 'd')); //depression

            if (tempPot > _synmax) tempPot = _synmax;
            if (tempDep < 0) tempDep = 0;

            //Potentiate G[k][spiker]
            if (_supcount[k] < _NSS[k] || (_supcount[k] == _NSS[k] && GPot >= _supthres)) {
                CheckThreshold(tempPot, k, spiker, 'a', 'p');
                CheckThreshold(tempPot, k, spiker, 's', 'p');
                _G(spiker, k) = tempPot;
            }

            //Depress G[spiker][k]
            if (_supcount[spiker] < _NSS[spiker] || (_supcount[spiker] == _NSS[spiker] && GDep >= _supthres)) {
                CheckThreshold(tempDep, spiker, k, 'a', 'd');
                CheckThreshold(tempDep, spiker, k, 's', 'd');
                _G(k, spiker) = tempDep;
            }
        }

    }
}

/**
 * PotentiationFunc
 *
 * @param time          Elapsed time of the trial.
 * @param spk_count     Size of the spk_times array.
 * @param spk_times     A log of the times the spiking neuron has previous spiked.
 * @param pd_type       'p' for potentiation, 'd' for depression
 */
double Synapses::PotentiationFunc( double time, int spk_count, double *spk_times, char pd_type ) {
    // ref L632
    double res = 0.0;
    double a;
    double delt, pwt, inv_dect;

    if (pd_type == 'p') {
        pwt = POTENTIATION_FUNT;
        inv_dect = INV_DECAY_LTP;
    } else if (pd_type == 'd') {
        pwt = DEPRESSION_FUNT;
        inv_dect = INV_DECAY_LTD;
    } else {
        // ERROR
        return 0.0;
    }

    for (int i = spk_count - 1; i >= 0; --i) {
        delt = time - spk_times[i];
        if (delt > _window) break;

        if (delt <= pwt) {
            a = delt / pwt;
        } else {
            a = exp(-(delt - pwt) * inv_dect);
        }

        res += a;
    }

    return res;
}

/**
 * Ensure that the synaptic strength of (pre, post) is within threshold limit.
 *
 * @param syn_str   The new synapse strength.
 * @param pre       Label of synapse start neuron.
 * @param post      Label of synapse end neuron.
 * @param syn_type  'a' for active, 's' for super
 * @param pd_type   'p' for potentiation, 'd' for depression
 */
void Synapses::CheckThreshold( double syn_str, int pre, int post, char syn_type, char pd_type ) {
    double thres;
    if (syn_type == 'a') thres = _actthres;
    else if (syn_type == 's') thres = _supthres;
    else return;// ERROR

    if (pd_type == 'p') {
        if (_G(post, pre) < thres && syn_str >= thres) {
            Activate(syn_type, pre, post);
        }
    } else if (pd_type == 'd') {
        if (_G(post, pre) >= thres && syn_str < thres) {
            Deactivate(syn_type, pre, post);
        }
    } else {
        // ERROR
        return;
    }
}

void Synapses::SynapticDecay() {
    // TODO: Verify indexing (it should be right considering the calls to CheckThreshold).
    for (int i = 0, pre = 0; pre < _size; ++pre) {
        for (int post = 0; post < _size; ++post, ++i) {
            CheckThreshold(_G[i] * _syndec, pre, post, 'a', 'd');
            CheckThreshold(_G[i] * _syndec, pre, post, 's', 'd');
            _G[i] *= _syndec;
        }
    }
}

double Synapses::GetPostSynapticLabel( char syn_type, int pre, bool *&post_arr ) {
    if (syn_type == 'a') {
        post_arr = _actsyn.row(pre);
        return _actcount[pre];
    } else { //  if(syn_type == 's')
        post_arr = _supsyn.row(pre);
        return _supcount[pre];
    }
}

int Synapses::CountSynapses(char syn_type) {
    int *cn_type;
    if(syn_type == 'a') {
        cn_type = _actcount;
    } else {
        cn_type = _supcount;
    }

    int sum = 0;
    for(int i = 0; i < _size; ++i) {
        sum += cn_type[i];
    }
    return sum;
}
