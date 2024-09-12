#ifndef SIGMOID_CPP
#define SIGMOID_CPP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "../core.h"
#include "../controller.h"
#include "../utils.h"
#include "sigmoid.h"

void SigmoidCalculate(DataStruct container){

    float* input = container.arg_f[0];
    float* output = container.arg_f[1];

    int64_t size = 1;
    for(int i=0; i<4; i++){
        size *= container.shape[0][i];
    }

    printf("Sig Calculate >> size : %d\r\n", size);

    for (int i = 0; i < size; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }

    
    printf("after sigmoid \r\n");
    printf("[input] \r\n");
    for(int i=0; i<10; i++){
        printf("%f ", *(input + size - i - 1));
    }
    printf("\r\n");
    printf("[output] \r\n");
    for(int i=0; i<10; i++){
        printf("%f ", *(output + size - i - 1));
    }
    printf("\r\n");

    return;
}

#endif