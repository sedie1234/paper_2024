#ifndef CORE_CPP
#define CORE_CPP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "controller.h"
#include "core.h"
#include "operations/conv.h"
#include "operations/sigmoid.h"
#include "operations/mul.h"
#include "operations/add.h"
#include "operations/sub.h"
#include "operations/div.h"
#include "operations/maxpool.h"
#include "operations/softmax.h"

extern SharedData *shared_data;
int core_status;

void runCore(int order){
    core_status = 1;
    Inst cmd;
    int i;
    for(i=0; i<8; i++){
        if(shared_data->instQueue[i].order == order){
            cmd = shared_data->instQueue[i];
            break;
        }
    }

    if(cmd.optype == CONVTYPE){
        runConv(cmd);
    }else if(cmd.optype == SIGMOIDTYPE){
        runSigmoid(cmd);
    }else if(cmd.optype == MULTYPE){
        runMul(cmd);
    }else if(cmd.optype == ADDTYPE){
        runAdd(cmd);
    }else if(cmd.optype == MAXPOOLTYPE){
        runMaxpool(cmd);
    }else if(cmd.optype == SOFTMAXTYPE){
        runSoftmax(cmd);
    }else if(cmd.optype == SUBTYPE){
        runSub(cmd);
    }else if(cmd.optype == DIVTYPE){
        runDiv(cmd);
    }else if(cmd.optype == FUSEDCONVTYPE){
        runFusedConv(cmd);
    }else{
        return;
    }

    shared_data->instQueue[i].order = -2;
    core_status = 0;
    return;
}

void runConv(Inst cmd){
    DataStruct container;
    
    //get bankinfo
    int inputbank = getBank(cmd.instID, 0);
    int input_startbank = (inputbank >> 16) & 0x0000FFFF;
    int input_endbank = inputbank & 0x0000FFFF;

    int kernelbank = getBank(cmd.instID, 1);
    int kernel_startbank = (kernelbank >>16) & 0x0000FFFF;
    int kernel_endbank = kernelbank & 0x0000FFFF;

    int biasbank = getBank(cmd.instID, 2);
    int bias_startbank = (biasbank >> 16) & 0x0000FFFF;
    int bias_endbank = biasbank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[input_startbank].shape, 32);
    memcpy(container.shape[1], shared_data->bankinfo[kernel_startbank].shape, 32);
    memcpy(container.shape[2], shared_data->bankinfo[bias_startbank].shape, 32);

    int64_t inputsize = getsize(container.shape[0], 4);
    int64_t kernelsize = getsize(container.shape[1], 4);
    int64_t biassize = getsize(container.shape[2], 4);


    float* inputbuffer = (float*)malloc(sizeof(float)*inputsize*1.01);
    float* kernelbuffer = (float*)malloc(sizeof(float)*kernelsize);
    float* biasbuffer = (float*)malloc(sizeof(float)*biassize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    container.config[0] = cmd.config[0];
    container.config[1] = cmd.config[1];
    container.config[2] = cmd.config[2];
    
    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);

    printf("outstartbank : %d \r\n", outstartbank);
    printf("allocatedBankNum : %d \r\n", outbanknum);

    //output bank lock and info set
    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
    
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[outstartbank + i].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[outstartbank + i].arg = 56;
        else
            shared_data->bankinfo[outstartbank + i].arg = 55;
        //out shape?
    }

    //read data
    char filename[20];

    bankLock(input_startbank, input_endbank-input_startbank+1);
    bankLock(kernel_startbank, kernel_endbank-kernel_startbank+1);
    bankLock(bias_startbank, bias_endbank-bias_startbank+1);

    //input read
    memset(filename, 0, 20);    
    int cnt=0;
    for(int i=input_startbank; i<input_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(inputsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), inputsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    //kernel read
    memset(filename, 0, 20);    
    cnt = 0;
    for(int i=kernel_startbank; i<kernel_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(kernelsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(kernelbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(kernelbuffer + cnt*BANKSIZE/4, sizeof(char), kernelsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    //bias read
    memset(filename, 0, 20);  
    cnt = 0;  
    for(int i=bias_startbank; i<bias_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(biassize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(biasbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(biasbuffer + cnt*BANKSIZE/4, sizeof(char), biassize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }
    //mapping data

    container.arg_f[0] = inputbuffer;
    container.arg_f[1] = kernelbuffer;
    container.arg_f[2] = biasbuffer;
    container.arg_f[3] = outbuffer;

    //calculate
    ConvCalculate(container);

    //savedata(output write)
    memset(filename, 0, 20);    
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    //arb bank release
    bankRelease(cmd.instID, 0);
    bankRelease(cmd.instID, 1);
    bankRelease(cmd.instID, 2);

    //bank unlock
    bankUnLock(input_startbank, input_endbank-input_startbank+1);
    bankUnLock(kernel_startbank, kernel_endbank-kernel_startbank+1);
    bankUnLock(bias_startbank, bias_endbank-bias_startbank+1);
    bankUnLock(outstartbank, outbanknum);
    
    free(inputbuffer);
    free(kernelbuffer);
    free(biasbuffer);
    free(outbuffer);
    
}

void runFusedConv(Inst cmd){
    DataStruct container;
    
    //get bankinfo
    int inputbank = getBank(cmd.instID, 0);
    int input_startbank = (inputbank >> 16) & 0x0000FFFF;
    int input_endbank = inputbank & 0x0000FFFF;

    int kernelbank = getBank(cmd.instID, 1);
    int kernel_startbank = (kernelbank >>16) & 0x0000FFFF;
    int kernel_endbank = kernelbank & 0x0000FFFF;

    int biasbank = getBank(cmd.instID, 2);
    int bias_startbank = (biasbank >> 16) & 0x0000FFFF;
    int bias_endbank = biasbank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[input_startbank].shape, 32);
    memcpy(container.shape[1], shared_data->bankinfo[kernel_startbank].shape, 32);
    memcpy(container.shape[2], shared_data->bankinfo[bias_startbank].shape, 32);

    int64_t inputsize = getsize(container.shape[0], 4);
    int64_t kernelsize = getsize(container.shape[1], 4);
    int64_t biassize = getsize(container.shape[2], 4);


    float* inputbuffer = (float*)malloc(sizeof(float)*inputsize*1.01);
    float* kernelbuffer = (float*)malloc(sizeof(float)*kernelsize);
    float* biasbuffer = (float*)malloc(sizeof(float)*biassize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    container.config[0] = cmd.config[0];
    container.config[1] = cmd.config[1];
    container.config[2] = cmd.config[2];
    
    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);

    printf("outstartbank : %d \r\n", outstartbank);
    printf("allocatedBankNum : %d \r\n", outbanknum);

    //output bank lock and info set
    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
    
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[outstartbank + i].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[outstartbank + i].arg = 56;
        else
            shared_data->bankinfo[outstartbank + i].arg = 55;
        //out shape?
    }

    //read data
    char filename[20];

    bankLock(input_startbank, input_endbank-input_startbank+1);
    bankLock(kernel_startbank, kernel_endbank-kernel_startbank+1);
    bankLock(bias_startbank, bias_endbank-bias_startbank+1);

    //input read
    memset(filename, 0, 20);    
    int cnt=0;
    for(int i=input_startbank; i<input_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(inputsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), inputsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    //kernel read
    memset(filename, 0, 20);    
    cnt = 0;
    for(int i=kernel_startbank; i<kernel_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(kernelsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(kernelbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(kernelbuffer + cnt*BANKSIZE/4, sizeof(char), kernelsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    //bias read
    memset(filename, 0, 20);  
    cnt = 0;  
    for(int i=bias_startbank; i<bias_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(biassize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(biasbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(biasbuffer + cnt*BANKSIZE/4, sizeof(char), biassize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }
    //mapping data

    container.arg_f[0] = inputbuffer;
    container.arg_f[1] = kernelbuffer;
    container.arg_f[2] = biasbuffer;
    container.arg_f[3] = outbuffer;

    //calculate
    FusedConvCalculate(container);

    //savedata(output write)
    memset(filename, 0, 20);    
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    //arb bank release
    bankRelease(cmd.instID, 0);
    bankRelease(cmd.instID, 1);
    bankRelease(cmd.instID, 2);

    //bank unlock
    bankUnLock(input_startbank, input_endbank-input_startbank+1);
    bankUnLock(kernel_startbank, kernel_endbank-kernel_startbank+1);
    bankUnLock(bias_startbank, bias_endbank-bias_startbank+1);
    bankUnLock(outstartbank, outbanknum);
    
    free(inputbuffer);
    free(kernelbuffer);
    free(biasbuffer);
    free(outbuffer);
    
}


void runSigmoid(Inst cmd){
    DataStruct container;
    printf("runSigmoid >> into runSigmoid\r\n");
    //get bankinfo
    int inputbank = getBank(cmd.instID, 0);
    printf("runSigmoid >> inputbank : %d\r\n", inputbank);
    if(inputbank == -1){
        printf("runSigmoid >> Fail to get inputbank : ");
        for(int j=0; j<16; j++){
            printf("%02X", cmd.instID.bytes[j]);
        }
        printf(" %d \r\n", cmd.arg);
    }


    int input_startbank = (inputbank >> 16) & 0x0000FFFF;
    int input_endbank = inputbank & 0x0000FFFF;



    memcpy(container.shape[0], shared_data->bankinfo[input_startbank].shape, 32);
    printf("runSigmoid >> memcpy complete\r\n");

    int64_t inputsize = getsize(container.shape[0], 4);

    printf("runSigmoid >> size : %d %d\r\n", inputsize, cmd.outsize);

    float* inputbuffer = (float*)malloc(sizeof(float)*inputsize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);  

    printf("runSigmoid >> out bank info : %d %d \r\n", outstartbank, outbanknum);

    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 

        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[i + outstartbank].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[i + outstartbank].arg = 56;
        else
            shared_data->bankinfo[i + outstartbank].arg = 55;
        //out shape?
    }  
    //read data
    char filename[20];
    bankLock(input_startbank, input_endbank-input_startbank+1);
    
    memset(filename, 0, 20);    
    int cnt=0;
    for(int i=input_startbank; i<input_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);

        printf("core read file : %s - remain data : %d", filename, inputsize*sizeof(float) - cnt*BANKSIZE);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(inputsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), inputsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        printf(" first data : %f %f", *(inputbuffer + cnt*BANKSIZE/4), *(inputbuffer + cnt*BANKSIZE/4 + 1));

        cnt++;
        fclose(mfile);
        printf("\r\n");
    }

    container.arg_f[0] = inputbuffer;
    container.arg_f[1] = outbuffer;


    printf("before sigmoid \r\n");
    printf("[=======input] \r\n");
    for(int i=0; i<10; i++){
        printf("%f ", *(inputbuffer + inputsize - 1 - i));
    }
    printf("\r\n");



    //calculate
    SigmoidCalculate(container);

    printf("[======out] \r\n");
    for(int i=0; i<10; i++){
        printf("%f ", *(outbuffer + inputsize - 1 - i));
    }
    printf("\r\n");


    //savedata(output write)
    memset(filename, 0, 20);    
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    bankRelease(cmd.instID, 1);

    bankUnLock(input_startbank, input_endbank-input_startbank+1);
    bankUnLock(outstartbank, outbanknum);

    free(inputbuffer);
    free(outbuffer);

}


void runMul(Inst cmd){
    DataStruct container;

    int Xbank = getBank(cmd.instID, 0);
    int X_startbank = (Xbank >> 16) & 0x0000FFFF;
    int X_endbank = Xbank & 0x0000FFFF;

    int Ybank = getBank(cmd.instID, 1);
    int Y_startbank = (Ybank >> 16) & 0x0000FFFF;
    int Y_endbank = Ybank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[X_startbank].shape, 32);
    memcpy(container.shape[1], shared_data->bankinfo[Y_startbank].shape, 32);

    int64_t Xsize = getsize(container.shape[0], 4);
    int64_t Ysize = getsize(container.shape[1], 4);

    float* Xbuffer = (float*)malloc(sizeof(float)*Xsize);
    float* Ybuffer = (float*)malloc(sizeof(float)*Ysize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);

    //output bank lock and info set
    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
    
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[outstartbank + i].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[outstartbank + i].arg = 56;
        else
            shared_data->bankinfo[outstartbank + i].arg = 55;
        //out shape?
    } 

    //read data
    char filename[20];

    bankLock(X_startbank, X_endbank-X_startbank+1);
    bankLock(Y_startbank, Y_endbank-Y_startbank+1);

    memset(filename, 0, 20);
    int cnt=0;
    for(int i=X_startbank; i<X_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Xsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), Xsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    memset(filename, 0, 20);
    cnt = 0;
    for(int i=Y_startbank; i<Y_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Ysize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), Ysize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    container.arg_f[0] = Xbuffer;
    container.arg_f[1] = Ybuffer;
    container.arg_f[2] = outbuffer;

    

    //calculate
    MulCalculate(container);

    //svaedata(output write)
    memset(filename, 0, 20);
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    bankRelease(cmd.instID, 0);
    bankRelease(cmd.instID, 1);

    bankUnLock(X_startbank, X_endbank-X_startbank+1);
    bankUnLock(Y_startbank, Y_endbank-Y_startbank+1);
    bankUnLock(outstartbank, outbanknum);

    free(Xbuffer);
    free(Ybuffer);
    free(outbuffer);

}

void runAdd(Inst cmd){
    DataStruct container;

    int Xbank = getBank(cmd.instID, 0);
    int X_startbank = (Xbank >> 16) & 0x0000FFFF;
    int X_endbank = Xbank & 0x0000FFFF;

    int Ybank = getBank(cmd.instID, 1);
    int Y_startbank = (Ybank >> 16) & 0x0000FFFF;
    int Y_endbank = Ybank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[X_startbank].shape, 32);
    memcpy(container.shape[1], shared_data->bankinfo[Y_startbank].shape, 32);

    int64_t Xsize = getsize(container.shape[0], 4);
    int64_t Ysize = getsize(container.shape[1], 4);

    float* Xbuffer = (float*)malloc(sizeof(float)*Xsize);
    float* Ybuffer = (float*)malloc(sizeof(float)*Ysize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);

    //output bank lock and info set
    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
    
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[outstartbank + i].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[outstartbank + i].arg = 56;
        else
            shared_data->bankinfo[outstartbank + i].arg = 55;
        //out shape?
    } 

    //read data
    char filename[20];

    bankLock(X_startbank, X_endbank-X_startbank+1);
    bankLock(Y_startbank, Y_endbank-Y_startbank+1);

    memset(filename, 0, 20);
    int cnt=0;
    for(int i=X_startbank; i<X_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Xsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), Xsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    memset(filename, 0, 20);
    cnt = 0;
    for(int i=Y_startbank; i<Y_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Ysize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), Ysize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    container.arg_f[0] = Xbuffer;
    container.arg_f[1] = Ybuffer;
    container.arg_f[2] = outbuffer;

    //calculate
    AddCalculate(container);

    //svaedata(output write)
    memset(filename, 0, 20);
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    bankRelease(cmd.instID, 0);
    bankRelease(cmd.instID, 1);

    bankUnLock(X_startbank, X_endbank-X_startbank+1);
    bankUnLock(Y_startbank, Y_endbank-Y_startbank+1);
    bankUnLock(outstartbank, outbanknum);

    free(Xbuffer);
    free(Ybuffer);
    free(outbuffer);

}

void runSub(Inst cmd){
    DataStruct container;

    int Xbank = getBank(cmd.instID, 0);
    int X_startbank = (Xbank >> 16) & 0x0000FFFF;
    int X_endbank = Xbank & 0x0000FFFF;

    int Ybank = getBank(cmd.instID, 1);
    int Y_startbank = (Ybank >> 16) & 0x0000FFFF;
    int Y_endbank = Ybank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[X_startbank].shape, 32);
    memcpy(container.shape[1], shared_data->bankinfo[Y_startbank].shape, 32);

    int64_t Xsize = getsize(container.shape[0], 4);
    int64_t Ysize = getsize(container.shape[1], 4);

    float* Xbuffer = (float*)malloc(sizeof(float)*Xsize);
    float* Ybuffer = (float*)malloc(sizeof(float)*Ysize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);

    //output bank lock and info set
    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
    
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[outstartbank + i].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[outstartbank + i].arg = 56;
        else
            shared_data->bankinfo[outstartbank + i].arg = 55;
        //out shape?
    } 

    //read data
    char filename[20];

    bankLock(X_startbank, X_endbank-X_startbank+1);
    bankLock(Y_startbank, Y_endbank-Y_startbank+1);

    memset(filename, 0, 20);
    int cnt=0;
    for(int i=X_startbank; i<X_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Xsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), Xsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    memset(filename, 0, 20);
    cnt = 0;
    for(int i=Y_startbank; i<Y_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Ysize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), Ysize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    container.arg_f[0] = Xbuffer;
    container.arg_f[1] = Ybuffer;
    container.arg_f[2] = outbuffer;

    //calculate
    SubCalculate(container);

    //svaedata(output write)
    memset(filename, 0, 20);
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    bankRelease(cmd.instID, 0);
    bankRelease(cmd.instID, 1);

    bankUnLock(X_startbank, X_endbank-X_startbank+1);
    bankUnLock(Y_startbank, Y_endbank-Y_startbank+1);
    bankUnLock(outstartbank, outbanknum);

    free(Xbuffer);
    free(Ybuffer);
    free(outbuffer);

}

void runDiv(Inst cmd){
    DataStruct container;

    int Xbank = getBank(cmd.instID, 0);
    int X_startbank = (Xbank >> 16) & 0x0000FFFF;
    int X_endbank = Xbank & 0x0000FFFF;

    int Ybank = getBank(cmd.instID, 1);
    int Y_startbank = (Ybank >> 16) & 0x0000FFFF;
    int Y_endbank = Ybank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[X_startbank].shape, 32);
    memcpy(container.shape[1], shared_data->bankinfo[Y_startbank].shape, 32);

    int64_t Xsize = getsize(container.shape[0], 4);
    int64_t Ysize = getsize(container.shape[1], 4);

    float* Xbuffer = (float*)malloc(sizeof(float)*Xsize);
    float* Ybuffer = (float*)malloc(sizeof(float)*Ysize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);

    //output bank lock and info set
    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
    
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[outstartbank + i].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[outstartbank + i].arg = 56;
        else
            shared_data->bankinfo[outstartbank + i].arg = 55;
        //out shape?
    } 

    //read data
    char filename[20];

    bankLock(X_startbank, X_endbank-X_startbank+1);
    bankLock(Y_startbank, Y_endbank-Y_startbank+1);

    memset(filename, 0, 20);
    int cnt=0;
    for(int i=X_startbank; i<X_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Xsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Xbuffer + cnt*BANKSIZE/4, sizeof(char), Xsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    memset(filename, 0, 20);
    cnt = 0;
    for(int i=Y_startbank; i<Y_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(Ysize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(Ybuffer + cnt*BANKSIZE/4, sizeof(char), Ysize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    container.arg_f[0] = Xbuffer;
    container.arg_f[1] = Ybuffer;
    container.arg_f[2] = outbuffer;

    //calculate
    DivCalculate(container);

    //svaedata(output write)
    memset(filename, 0, 20);
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    bankRelease(cmd.instID, 0);
    bankRelease(cmd.instID, 1);

    bankUnLock(X_startbank, X_endbank-X_startbank+1);
    bankUnLock(Y_startbank, Y_endbank-Y_startbank+1);
    bankUnLock(outstartbank, outbanknum);

    free(Xbuffer);
    free(Ybuffer);
    free(outbuffer);

}

void runMaxpool(Inst cmd){
    DataStruct container;
    
    //get bankinfo
    int inputbank = getBank(cmd.instID, 0);
    int input_startbank = (inputbank >> 16) & 0x0000FFFF;
    int input_endbank = inputbank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[input_startbank].shape, 32);

    int64_t inputsize = getsize(container.shape[0], 4);

    float* inputbuffer = (float*)malloc(sizeof(float)*inputsize*1.01);

    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    container.config[0] = cmd.config[0];
    container.config[1] = cmd.config[1];
    container.config[2] = cmd.config[2];
    
    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);

    //output bank lock and info set
    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
    
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[outstartbank + i].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[outstartbank + i].arg = 56;
        else
            shared_data->bankinfo[outstartbank + i].arg = 55;
        //out shape?
    }

    //read data
    char filename[20];

    bankLock(input_startbank, input_endbank-input_startbank+1);

    //input read
    memset(filename, 0, 20);    
    int cnt=0;
    for(int i=input_startbank; i<input_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(inputsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), inputsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }
        cnt++;
        fclose(mfile);
    }

    //mapping data
    container.arg_f[0] = inputbuffer;
    
    container.arg_f[1] = outbuffer;

    //calculate
    MaxpoolCalculate(container);

    //savedata(output write)
    memset(filename, 0, 20);    
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    //arb bank release
    bankRelease(cmd.instID, 0);
    bankRelease(cmd.instID, 1);
    bankRelease(cmd.instID, 2);

    //bank unlock
    bankUnLock(input_startbank, input_endbank-input_startbank+1);
    bankUnLock(outstartbank, outbanknum);
    
    free(inputbuffer);
    free(outbuffer);
    
}

void runSoftmax(Inst cmd){
    DataStruct container;
    //get bankinfo
    int inputbank = getBank(cmd.instID, 0);

    int input_startbank = (inputbank >> 16) & 0x0000FFFF;
    int input_endbank = inputbank & 0x0000FFFF;

    memcpy(container.shape[0], shared_data->bankinfo[input_startbank].shape, 32);

    int64_t inputsize = getsize(container.shape[0], 4);

    float* inputbuffer = (float*)malloc(sizeof(float)*inputsize);
    float* outbuffer = (float*)malloc(sizeof(float)*cmd.outsize);

    //alloc output mem
    int outstartbank = seekBank(cmd.outsize);
    int outbanknum = allocatedBankNum(cmd.outsize);  

    for(int i=0; i<outbanknum; i++){
        int bankIndex = outstartbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 

        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;

        shared_data->bankinfo[i + outstartbank].instID = cmd.instID;
        if(i==0)
            shared_data->bankinfo[i + outstartbank].arg = 56;
        else
            shared_data->bankinfo[i + outstartbank].arg = 55;
        //out shape?
    }  
    //read data
    char filename[20];
    bankLock(input_startbank, input_endbank-input_startbank+1);
    
    memset(filename, 0, 20);    
    int cnt=0;
    for(int i=input_startbank; i<input_endbank+1; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);

        FILE* mfile = fopen(filename, "rb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(inputsize*sizeof(float) - cnt*BANKSIZE > BANKSIZE){
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fread(inputbuffer + cnt*BANKSIZE/4, sizeof(char), inputsize*sizeof(float) - cnt*BANKSIZE, mfile);
        }

        cnt++;
        fclose(mfile);
        printf("\r\n");
    }

    container.arg_f[0] = inputbuffer;
    container.arg_f[1] = outbuffer;

    //calculate
    SoftmaxCalculate(container);

    //savedata(output write)
    memset(filename, 0, 20);    
    for(int i=0; i<outbanknum; i++){
        sprintf(filename, "%s%d", BANK_BASE_FILENAME, outstartbank + i);
        FILE* mfile;
        mfile = fopen(filename, "wb");
        if(mfile == NULL){
            perror("Error opening file for writing");
            return;
        }
        if(cmd.outsize - i*BANKSIZE > BANKSIZE){
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), BANKSIZE, mfile);
        }else{
            fwrite(outbuffer + i*BANKSIZE/4, sizeof(char), cmd.outsize - i*BANKSIZE, mfile);
        }
        fclose(mfile);
    }

    bankRelease(cmd.instID, 1);

    bankUnLock(input_startbank, input_endbank-input_startbank+1);
    bankUnLock(outstartbank, outbanknum);

    free(inputbuffer);
    free(outbuffer);

}


int64_t getsize(int64_t* shape, int rank){
    int64_t size=1;
    for(int i=0; i<rank; i++){
        size *= shape[i];
    }
    return size;
}
#endif