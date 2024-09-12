#ifndef CORE_H
#define CORE_H


#include "controller.h"

enum{
    NONETYPE = -1,
    CONVTYPE = 1,
    SIGMOIDTYPE,
    MULTYPE,
    ADDTYPE,
    MAXPOOLTYPE,
    SOFTMAXTYPE,
    SUBTYPE,
    DIVTYPE
};


typedef struct {
    float* arg_f[10];
    int* arg_i[10];
    int config[10];
    int64_t shape[5][4];
}DataStruct;

void runCore(int order);

void runConv(Inst cmd);
void runSigmoid(Inst cmd);
void runMul(Inst cmd);
void runAdd(Inst cmd);
void runSub(Inst cmd);
void runDiv(Inst cmd);
void runMaxpool(Inst cmd);
void runSoftmax(Inst cmd);
int64_t getsize(int64_t* shape, int rank);

#endif