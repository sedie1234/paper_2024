#ifndef CONV_CPP
#define CONV_CPP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "../core.h"
#include "../controller.h"
#include "../utils.h"
#include "conv.h"

void nhwc_to_nchw(float* nhwc_input, float* nchw_output, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    // NHWC: [N, H, W, C] -> NCHW: [N, C, H, W]
                    nchw_output[n * C * H * W + c * H * W + h * W + w] =
                        nhwc_input[n * H * W * C + h * W * C + w * C + c];
                }
            }
        }
    }
}

void kernel_nhwc_to_nchw(float* nhwc_kernel, float* nchw_kernel, int M, int C, int KH, int KW) {
    for (int m = 0; m < M; m++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                for (int c = 0; c < C; c++) {
                    // NHWC Kernel: [M, KH, KW, C] -> NCHW Kernel: [M, C, KH, KW]
                    nchw_kernel[m * C * KH * KW + c * KH * KW + kh * KW + kw] =
                        nhwc_kernel[m * KH * KW * C + kh * KW * C + kw * C + c];
                }
            }
        }
    }
}

void nchw_to_nhwc(float* nchw_input, float* nhwc_output, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    // NCHW: [N, C, H, W] -> NHWC: [N, H, W, C]
                    nhwc_output[n * H * W * C + h * W * C + w * C + c] =
                        nchw_input[n * C * H * W + c * H * W + h * W + w];
                }
            }
        }
    }
}

void kernel_nchw_to_nhwc(float* nchw_kernel, float* nhwc_kernel, int M, int C, int KH, int KW) {
    for (int m = 0; m < M; m++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                for (int c = 0; c < C; c++) {
                    // NCHW Kernel: [M, C, KH, KW] -> NHWC Kernel: [M, KH, KW, C]
                    nhwc_kernel[m * KH * KW * C + kh * KW * C + kw * C + c] =
                        nchw_kernel[m * C * KH * KW + c * KH * KW + kh * KW + kw];
                }
            }
        }
    }
}

void ConvCalculate(DataStruct container) {
    // Input parameters
    float* input = container.arg_f[0];
    float* kernel = container.arg_f[1];
    float* bias = container.arg_f[2];
    float* output = container.arg_f[3];
    int kernelSize = container.config[0];
    int padding = container.config[1];
    int stride = container.config[2];

    int64_t* input_shape = container.shape[0];    // [N, C, H, W]
    int64_t* kernel_shape = container.shape[1];   // [M, C, KH, KW]
    int64_t* bias_shape = container.shape[2];     // [M]

    int N = input_shape[0];   // Batch size
    int C = input_shape[1];   // Input channels
    int H = input_shape[2];   // Input height
    int W = input_shape[3];   // Input width

    int M = kernel_shape[0];  // Output channels
    int KH = kernel_shape[2]; // Kernel height
    int KW = kernel_shape[3]; // Kernel width

    // Output dimensions
    int outH = (H - KH + 2 * padding) / stride + 1;
    int outW = (W - KW + 2 * padding) / stride + 1;

    int max = 0;

    float* input_nchw = (float*)malloc(N*C*H*W*sizeof(float));
    float* kernel_nchw = (float*)malloc(M*C*KH*KW*sizeof(float));

    nhwc_to_nchw(input, input_nchw, N, C, H, W);
    // kernel_nhwc_to_nchw(kernel, kernel_nchw, M, C, KH, KW);

    memcpy(input, input_nchw, N*H*W*C*sizeof(float));
    // memcpy(kernel, kernel_nchw, M*KH*KW*C*sizeof(float));

    // Convolution operation
    for (int n = 0; n < N; n++) {          // Iterate over batch
        for (int m = 0; m < M; m++) {      // Iterate over output channels
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    float sum = 0.0f;

                    for (int c = 0; c < C; c++) {  // Iterate over input channels
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                int h_in = h * stride + kh - padding;
                                int w_in = w * stride + kw - padding;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    sum += input[n * C * H * W + c * H * W + h_in * W + w_in] *
                                           kernel[m * C * KH * KW + c * KH * KW + kh * KW + kw];

                                    if(max < n * C * H * W + c * H * W + h_in * W + w_in)
                                        max = n * C * H * W + c * H * W + h_in * W + w_in;
                                }
                            }
                        }
                    }
                    // Add bias and store the result
                    output[n * M * outH * outW + m * outH * outW + h * outW + w] = sum + bias[m];
                }
            }
        }
    }



    printf("\r\nConv >>> [=======input] \r\n");
    for(int i=0; i<20; i++){
        printf("%f ", *(input + i));
    }
    printf("\r\n");

    printf("\r\nConv >>> [=======kernel] \r\n");
    for(int i=0; i<20; i++){
        printf("%f ", *(kernel + i));
    }
    printf("\r\n");

    printf("\r\nConv >>> [=======output] \r\n");
    for(int i=0; i<20; i++){
        printf("%f ", *(output + i));
    }
    printf("\r\n");

    free(input_nchw);
    free(kernel_nchw);
}


void FusedConvCalculate(DataStruct container){

    // Input parameters
    float* input = container.arg_f[0];
    float* kernel = container.arg_f[1];
    float* bias = container.arg_f[2];
    float* output = container.arg_f[3];
    int kernelSize = container.config[0];
    int padding = container.config[1];
    int stride = container.config[2];

    int64_t* input_shape = container.shape[0];    // [N, C, H, W]
    int64_t* kernel_shape = container.shape[1];   // [M, C, KH, KW]
    int64_t* bias_shape = container.shape[2];     // [M]

    int N = input_shape[0];   // Batch size
    int C = input_shape[1];   // Input channels
    int H = input_shape[2];   // Input height
    int W = input_shape[3];   // Input width

    int M = kernel_shape[0];  // Output channels
    int KH = kernel_shape[2]; // Kernel height
    int KW = kernel_shape[3]; // Kernel width

    // Output dimensions
    int outH = (H - KH + 2 * padding) / stride + 1;
    int outW = (W - KW + 2 * padding) / stride + 1;

    int max = 0;

    float* input_nchw = (float*)malloc(N*C*H*W*sizeof(float));
    float* kernel_nchw = (float*)malloc(M*C*KH*KW*sizeof(float));

    int64_t outsize = N*M*outH*outW;
    float* conv_output = (float*)malloc(outsize*sizeof(float));    
    float* sigmoid_output = (float*)malloc(outsize*sizeof(float));    

    nhwc_to_nchw(input, input_nchw, N, C, H, W);
    // kernel_nhwc_to_nchw(kernel, kernel_nchw, M, C, KH, KW);

    memcpy(input, input_nchw, N*H*W*C*sizeof(float));
    // memcpy(kernel, kernel_nchw, M*KH*KW*C*sizeof(float));

    // Convolution operation
    for (int n = 0; n < N; n++) {          // Iterate over batch
        for (int m = 0; m < M; m++) {      // Iterate over output channels
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    float sum = 0.0f;

                    for (int c = 0; c < C; c++) {  // Iterate over input channels
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                int h_in = h * stride + kh - padding;
                                int w_in = w * stride + kw - padding;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    sum += input[n * C * H * W + c * H * W + h_in * W + w_in] *
                                           kernel[m * C * KH * KW + c * KH * KW + kh * KW + kw];

                                    if(max < n * C * H * W + c * H * W + h_in * W + w_in)
                                        max = n * C * H * W + c * H * W + h_in * W + w_in;
                                }
                            }
                        }
                    }
                    // Add bias and store the result
                    conv_output[n * M * outH * outW + m * outH * outW + h * outW + w] = sum + bias[m];
                }
            }
        }
    }

    for (int i = 0; i < outsize; i++) {
        sigmoid_output[i] = 1.0f / (1.0f + expf(-conv_output[i]));
    }

    for (int i=0; i<outsize; i++){
        output[i] = sigmoid_output[i]*conv_output[i];
    }


    free(input_nchw);
    free(kernel_nchw);
    free(conv_output);
    free(sigmoid_output);
}

#endif