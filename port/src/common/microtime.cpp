#include "microtime.h"
#include <Windows.h>

#define TEST 0 /* Set to 0 except when testing the code or timer resolution */

double get_microtime_resolution(void) {
    double time1, time2;

    time1 = microtime();
    do {
        time2 = microtime();
    } while (time1 == time2);

    return time2 - time1;
}
static bool initialized = false;
static LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
static LARGE_INTEGER Frequency;

double microtime(void) {
	if (initialized == false) {
		QueryPerformanceFrequency( &Frequency );
		QueryPerformanceCounter( &StartingTime );
		initialized = true;
	}

	QueryPerformanceCounter( &EndingTime );
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	//
	// We now have the elapsed number of ticks, along with the
	// number of ticks-per-second. We use these values
	// to convert to the number of elapsed microseconds.
	// To guard against loss-of-precision, we convert
	// to microseconds *before* dividing by ticks-per-second.
	//

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

    return (double) ElapsedMicroseconds.QuadPart;
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
