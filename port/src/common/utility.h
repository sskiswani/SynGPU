#ifndef PORT_UTILITY_H
#define PORT_UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include "microtime.h"

//#define LOG(format, ...) printf("********** {{ " format " }} **********\n", ##__VA_ARGS__);
#define LOG(format, ...)

struct Timer {
    Timer() { }

    void Start() {
        _start = microtime();
    }

    double Stop() {
        _stop = microtime();
        return (_stop - _start);
    }

    double Duration() {
        return (_stop - _start);
    }

private:
    double _start, _stop;
};


#endif //PORT_UTILITY_H
