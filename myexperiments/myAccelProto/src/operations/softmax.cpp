#ifndef SOFTMAX_CPP
#define SOFTMAX_CPP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "../core.h"
#include "../controller.h"
#include "../utils.h"
#include "softmax.h"

void SoftmaxCalculate(DataStruct container){

    float* input = container.arg_f[0];
    float* output = container.arg_f[1];

    int64_t size = 1;
    for(int i=0; i<4; i++){
        size *= container.shape[0][i];
    }
    
    int C = container.shape[0][1];  // C: 채널 (Softmax 축)
    int H = container.shape[0][2];  // H: 높이
    int W = container.shape[0][3];  // W: 너비

    // Softmax 연산
    for (int n = 0; n < container.shape[0][0]; ++n) {  // 배치 크기 N
        for (int h = 0; h < H; ++h) {                   // 높이 H
            for (int w = 0; w < W; ++w) {               // 너비 W
                // Step 1: Compute exp for each element in the channel
                float sum_exp = 0.0f;
                for (int c = 0; c < C; ++c) {
                    int index = ((n * C + c) * H + h) * W + w;
                    output[index] = expf(input[index]);  // 입력 값을 지수화
                    sum_exp += output[index];            // 지수화 값의 합
                }

                // Step 2: Normalize by the sum of the exponents
                for (int c = 0; c < C; ++c) {
                    int index = ((n * C + c) * H + h) * W + w;
                    output[index] /= sum_exp;            // 각 값을 지수 합으로 나눔
                }
            }
        }
    }

    return;
}

#endif