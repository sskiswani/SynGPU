#ifndef _microtime_h_
#define _microtime_h_

#ifdef __cplusplus
extern "C" {
#endif

  double    microtime(void); /* Time in micro-seconds */
  double    get_microtime_resolution(void); /* Timer resolution in micro-seconds */

#ifdef __cplusplus
}
#endif


#endif