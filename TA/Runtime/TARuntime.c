#include <pthread.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


void file_write(float *data){
    printf("test : file_write\r\n");
    for(int i=0; i<6; i++){
        printf("%f ", data[i]);
    }
    printf("\r\n");
    return;
}

float* file_read(){
    // float val[6] = {0, 1, 2, 3, 4, 5};
    float* val = (float*)malloc(6*sizeof(float));
    for(int i=0; i<6; i++){
        val[i] = i*0.1;
    }
    printf("test : file_read\r\n");
    return val;
}

void OMInitAccelTA() {
    printf("OMInitAccelTA!\r\n");
    return;
}


uint64_t OMInitCompatibleAccelTA(uint64_t versionNum){
    printf("Compatible TA Accelerator Init!!\r\n");
    return -1;
}


