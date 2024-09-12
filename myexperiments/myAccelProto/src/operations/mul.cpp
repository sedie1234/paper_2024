#ifndef MUL_CPP
#define MUL_CPP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "../core.h"
#include "../controller.h"
#include "../utils.h"
#include "mul.h"

void MulCalculate(DataStruct container){


    float* X = container.arg_f[0];
    float* Y = container.arg_f[1];
    float* output = container.arg_f[2];

    int64_t size = 1;
    for(int i=0; i<4; i++){
        size *= container.shape[0][i];
    }

    for (int i = 0; i < size; i++) {
        output[i] = X[i] * Y[i];
    }

}

#endif