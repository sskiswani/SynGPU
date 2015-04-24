#ifndef _microtime_h_
#define _microtime_h_

#define US_TO_S(x) ((x) / 1e6)
#define US_TO_MS(x) ((x) / 1e3)

#ifdef __cplusplus
extern "C" {
#endif

  double microtime(void); /* Time in micro-seconds */
  double get_microtime_resolution(void); /* Timer resolution in micro-seconds */

#ifdef __cplusplus
}
#endif

#endif
