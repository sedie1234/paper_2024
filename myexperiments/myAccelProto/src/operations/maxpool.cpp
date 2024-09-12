#ifndef MAXPOOL_CPP
#define MAXPOOL_CPP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

#include "../core.h"
#include "../controller.h"
#include "../utils.h"
#include "maxpool.h"

void MaxpoolCalculate(DataStruct container){


    float* X = container.arg_f[0];
    float* output = container.arg_f[1];
    
    int kernelSize = container.config[0];
    int padding = container.config[1];
    int stride = container.config[2];

    int64_t* input_shape = container.shape[0];    // [N, C, H, W]

    int N = input_shape[0];   // Batch size
    int C = input_shape[1];   // Input channels
    int H = input_shape[2];   // Input height
    int W = input_shape[3];   // Input width


    // 출력 텐서의 크기를 계산합니다.
    int outH = (H + 2 * padding - kernelSize) / stride + 1;
    int outW = (W + 2 * padding - kernelSize) / stride + 1;

    // MaxPool 연산 수행
    for (int n = 0; n < N; ++n) { // Batch
        for (int c = 0; c < C; ++c) { // Channels
            for (int h = 0; h < outH; ++h) { // Output height
                for (int w = 0; w < outW; ++w) { // Output width
                    float maxVal = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kernelSize; ++kh) { // Kernel height
                        for (int kw = 0; kw < kernelSize; ++kw) { // Kernel width
                            int inH = h * stride + kh - padding;
                            int inW = w * stride + kw - padding;
                            if (inH >= 0 && inH < H && inW >= 0 && inW < W) {
                                int inputIndex = ((n * C + c) * H + inH) * W + inW;
                                maxVal = std::max(maxVal, X[inputIndex]);
                            }
                        }
                    }
                    int outputIndex = ((n * C + c) * outH + h) * outW + w;
                    output[outputIndex] = maxVal;
                }
            }
        }
    }

}

#endif