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
                    double eq_syn) {
    _size = network_size;
    _window = window;
    _actthres = act_thres;
    _supthres = sup_thres;
    _synmax = syn_max;
    _syndec = syn_decay;
    _GLTP = eq_syn;

    // Allocate data
    _G = new double[_size * _size];

    _actcount = new int[_size];
    _supcount = new int[_size];

    _actsyn = new int *[_size];
    _supsyn = new int *[_size];

    _NSS = new int[_size];
    for (int i = 0; i < _size; ++i) {
        _NSS[i] = tempNSS;
    }

    if (form_opt == 1) {
        // Randomly activate fract_act of all synapses
        for (int y = 0, i = 0; y < _size; ++y) {
            _actcount[y] = 0;
            _supcount[y] = 0;
            for (int x = 0; x < _size; ++x, ++i) {
                if (x == y) { // self synapses not allowed.
                    _G[i] = 0;
                    continue;
                }

                if (ran1(&seed) <= fract_act) {
                    _G[i] = _actthres + (_synmax - _actthres) * ran1(&seed);
                    Activate('a', y, x);
                } else {
                    _G[i] = _actthres * ran1(&seed);
                }
                if (_G[i] > _synmax) _G[i] = _synmax;
            }
        }
    } else {
        // Default initialization.
        for (int i = 0; i < _size; ++i) {
            _actcount[i] = 0;
            _supcount[i] = 0;
            for (int j = 0; j < _size; ++j) {
                _G[i * _size + j] = glob;
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
 */
void Synapses::Activate( char p, int pre, int post ) {
    // 'a' for active, 's' for super
    int *l, *m;
    if (p == 'a') {
        l = &_actcount[pre]; // l points to # of active synapses of pre.
        // TODO: Handle reallocation of old supcount and supsyn.
        m = _actsyn[pre];
    } else if (p == 's') {
        l = &_supcount[pre];
        // TODO: Handle reallocation of old supcount and supsyn.
        m = _supsyn[pre];
    } else {
        // Invalid argument
        return;
    }

    m[(*l)] = post; // Place new synapses here.
    ++(*l);         // Increment counter.
}

void Synapses::Deactivate( char p, int pre, int post ) {
    // 'a' for active, 's' for super
    int *temp, *l, *m;
    if (p == 'a') {
        l = &_actcount[pre]; // l points to # of active synapses of pre.
        // TODO: Handle reallocation of old supcount and supsyn.
        m = _actsyn[pre];
    } else if (p == 's') {
        l = &_supcount[pre];
        // TODO: Handle reallocation of old supcount and supsyn.
        m = _supsyn[pre];
    } else {
        // Invalid argument
        return;
    }

    if ((*l) != 1) { // If the last post is not being deactivated...
        // TODO: Need to handle the shrinking/increasing values.
        // Don't include post when placing values in temp back into supsyn[pre], so move the last value into its place
        for (int i = 0; i < (*l); ++i) {
            if (temp[i] == post) {
                temp[i] = temp[(*l) - 1];
                break;
            }
        }

        --(*l);

        for (int i = 0; i < (*l); ++i) {
            m[i] = temp[i];
        }

        delete[] temp;
    } else { // Last post is being deactivated.
        --(*l);
    }
}

void Synapses::Synaptic_Plasticity( int spiker, double t, Neuron **net ) {
    double *spk_times;
    int spk_count;
    double tempPot, tempDep, GPot, GDep;

    int g_idx = spiker * _size;
    for (int k = 0; k < _size; ++k) {
        if (k == spiker) continue;

        GPot = _G[k * _size + spiker];
        GDep = _G[g_idx + k];
        // TODO: Access spike history of neuron.
//        spk_count = net[k]->get_spkhist(spk_times);
        if (spk_count != 0 && spk_times[spk_count - 1] + _window >= t) {
            tempPot = GPot + AMPLITUDE_LTP * _GLTP * PotentiationFunc(t, spk_count, spk_times, 'p'); //potentiation
            tempDep = GDep * (1 - AMPLITUDE_LTP * PotentiationFunc(t, spk_count, spk_times, 'd')); //depression

            if (tempPot > _synmax) tempPot = _synmax;
            if (tempDep < 0) tempDep = 0;

            //Potentiate G[k][spiker]
            if (_supcount[k] < _NSS[k] || (_supcount[k] == _NSS[k] && GPot >= _supthres)) {
                CheckThreshold(tempPot, k, spiker, 'a', 'p');
                CheckThreshold(tempPot, k, spiker, 's', 'p');
                _G[k * _size + spiker] = tempPot;
            }

            //Depress G[spiker][k]
            if (_supcount[spiker] < _NSS[spiker] || (_supcount[spiker] == _NSS[spiker] && GDep >= _supthres)) {
                CheckThreshold(tempDep, spiker, k, 'a', 'd');
                CheckThreshold(tempDep, spiker, k, 's', 'd');
                _G[g_idx + k] = tempDep;
            }
        }
    }
}

double Synapses::PotentiationFunc( double time, int spsc, double *hist, char pd_type ) {
    // ref L632
    // p=='p' for potentiation, p=='d' for depression
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

    for (int i = spsc - 1; i >= 0; --i) {
        delt = time - hist[i];
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

void Synapses::CheckThreshold( double syn_str, int pre, int post, char syn_type, char pd_type ) {
    //p=='a'/p=='s', check if G crossed active/super threshold
    //q == 'p' for potentiation, q == 'd' for depression
    double thres;
    if (syn_type == 'a') thres = _actthres;
    else if (pd_type == 's') thres = _supthres;
    else return;// ERROR

    if (pd_type == 'p') {
        if (_G[pre * _size + post] < thres && syn_str >= thres) {
            Activate(syn_type, pre, post);
        }
    } else if (pd_type == 'd') {
        if (_G[pre * _size * post] >= thres && syn_str < thres) {
            Deactivate(syn_type, pre, post);
        }
    } else {
        // ERROR
        return;
    }
}

void Synapses::SynapticDecay() {
    for (int i = 0, y = 0; y < _size; ++y) {
        for (int x = 0; x < _size; ++x, ++i) {
            CheckThreshold(_G[i] * _syndec, y, x, 'a', 'd');
            CheckThreshold(_G[i] * _syndec, y, x, 's', 'd');
            _G[i] *= _syndec;
        }
    }
}

double Synapses::GetPostSynapticLabel( char syn_type, int pre, int *&post ) {
    if(syn_type == 'a') {
        post = _actsyn[pre];
        return _actcount[pre];
    } else { //  if(syn_type == 's')
        post = _supsyn[pre];
        return _supcount[pre];
    }
}
