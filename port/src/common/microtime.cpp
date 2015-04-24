#include <sys/time.h>
#include "microtime.h"

#define TEST 0 /* Set to 0 except when testing the code or timer resolution */

double get_microtime_resolution(void) {
    double time1, time2;

    time1 = microtime();
    do {
        time2 = microtime();
    } while (time1 == time2);

    return time2 - time1;
}

double microtime(void) {
    struct timeval t;

    gettimeofday(&t, 0);

    return 1.0e6 * t.tv_sec + (double) t.tv_usec;
}


#if TEST == 1

#include <stdio.h>
int main(void) {
  unsigned long i, n=100000000;
  double result = 0.0, time1, time2;

  printf("Timer resolution = %g micro seconds\n", get_microtime_resolution());

  time1 = microtime();
  for(i = 1; i < n; i++)
    result += 1.0 / i;
  time2 = microtime();

  printf("Time taken = %g seconds\n", (time2-time1)/1.0e6);

  return 0;
}

#endif
