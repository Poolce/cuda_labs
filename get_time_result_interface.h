#ifndef GET_TIME_RESULT_INTERFACE__H
#define GET_TIME_RESULT_INTERFACE__H
#include "taxpy_interface.h"
#include <cuda_runtime.h>

timeResult GetDResultEXP(int vectorSize, int Xinc, int Yinc, double alpfa);

timeResult GetFResultEXP(int vectorSize, int Xinc, int Yinc, float alpfa);


#endif //GET_TIME_RESULT_INTERFACE__H