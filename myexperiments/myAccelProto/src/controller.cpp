#ifndef CONTROLLER_CPP
#define CONTROLLER_CPP

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include "controller.h"
#include "utils.h"

// extern SharedData *shared_data;

SharedData *shared_data;
QueueManager qm;

void signal_handler(int signum) {
    printf("Reader: Received signal %d\n", signum);

    sigset_t mask, oldmask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    sigprocmask(SIG_BLOCK, &mask, &oldmask);

    if(shared_data->interrupt_class == SIG_WRITE){
        //sig_write
        sigWriteCallback();
    }else if(shared_data->interrupt_class == SIG_READ){
        //sig_read
        sigReadCallback();
    }else if(shared_data->interrupt_class == SIG_RELEASE){
        //sig_release
        sigReleaseCallback();
    }else{
        return;
    }
    
}

void sharedDataInit(){
    for(int i=0; i<8; i++){
        shared_data->instQueue[i].order = -1;
        shared_data->instQueue[i].optype = -1;
    }
    
    shared_data->requested_inst.optype = -1;
    for(int i=0; i<BANKNUM; i++){
        shared_data->bankinfo[i].arg = -1;
    }
    shared_data->GBlockinfo = 0;
    memset(shared_data->banklockinfo, 0, sizeof(shared_data->banklockinfo));

    printf("shared data inited\r\n");

    return;
}

bool checkInstPush(){
    if(shared_data->requested_inst.optype == -1){
        return false;
    }

    return true;
}

void sigWriteCallback(){

    while(true){
        if((((shared_data->GBlockinfo)&H2C_MASK) == 0)){
    //H2C file lock
            shared_data->GBlockinfo|=H2C_MASK;
            
            FILE *file;
            FILE *mfile;
            unsigned char configbuffer[64];
            size_t bytesRead;
            char* buffer;

            MD5_t MD5;
            int datasize;
            int8_t usage;
            int64_t shape[4];
    //header read
            file = fopen(H2C_FILENAME, "rb");
            if(file == NULL){
                perror("Error opening file");
                return;
            }

            bytesRead = fread(configbuffer, 1, sizeof(configbuffer), file);
            memcpy(&(MD5.bytes[0]), &(configbuffer[0]), 16);
            memcpy(&datasize, &(configbuffer[16]), 4);
            memcpy(&usage, &(configbuffer[20]), 1);
            memcpy(shape, &(configbuffer[21]), 32);

            printf("usage : %X \r\n", usage);


    //data read        
            buffer = (char*) malloc(datasize);

            bytesRead = fread(buffer, 1, datasize, file);
    
            printf("buffer %d :", bytesRead);
            for(int i=0; i<10; i++){
                printf(" %f", *(float *)(buffer + datasize - 4 - 4*i));
            }
            printf("\r\n");

            int startbank = seekBank(datasize);
            int banknum = allocatedBankNum(datasize);
    //bank lock
            for(int i=0; i<banknum; i++){
                int bankIndex = startbank + i;
                int arrayIndex = bankIndex / 32;  
                int bitPosition = bankIndex % 32; 
                
                if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
                    shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
                else
                    i--;

                // shared_data->bankinfo[startbank+i].instID = MD5;

                memcpy(&(shared_data->bankinfo[startbank+i].instID.bytes[0]), &(MD5.bytes[0]), 16);
                shared_data->bankinfo[startbank+i].arg = usage;
                memcpy(shared_data->bankinfo[startbank+i].shape, shape, 32);

            }
    //data write to bank
            char filename[20];
            printf("datasize : %d\r\n", datasize);
            for(int i=0; i<banknum; i++){
                sprintf(filename, "%s%d", BANK_BASE_FILENAME, startbank + i);

#if _PRINT_
                printf("writeCallback : filename = %s\r\n", filename);
#endif
                mfile = fopen(filename, "wb");
                if(mfile == NULL){
                    perror("Error opening file for writing");
                    return;
                }

                if(datasize - i*BANKSIZE > BANKSIZE){
                    fwrite(buffer + i*BANKSIZE, sizeof(char), BANKSIZE, mfile);
                }else{
                    fwrite(buffer + i*BANKSIZE, sizeof(char), datasize - i*BANKSIZE, mfile);
                }
                fclose(mfile);
            }
    //bank unlock
            for(int i=0; i<banknum; i++){
                int bankIndex = startbank + i;
                int arrayIndex = bankIndex / 32;  
                int bitPosition = bankIndex % 32; 
                
                shared_data->banklockinfo[arrayIndex] &= (0 << bitPosition);
            }
    //H2C unlock
            shared_data->GBlockinfo &= ~H2C_MASK;
            shared_data->GBlockinfo &= ~H2C_CLEARMASK;
    //free, close
            free(buffer);
            fclose(file);
            return;
        }
    }
    return;
}

void sigReadCallback(){

    while(true){
        if(((shared_data->GBlockinfo)&C2H_MASK) == 0){
    
    //C2H file lock
            shared_data->GBlockinfo|=C2H_MASK;

            FILE *file;
            FILE *mfile;
            char configbuffer[64];
            size_t bytesRead;
            char* buffer;

            MD5_t MD5;
            int datasize;
            int output_bankinfo;
            int output_startbank;
            int output_endbank;
    //header read
            file = fopen(C2H_FILENAME, "rb");
            if(file == NULL){
                perror("Error opening file");
                return;
            }

            bytesRead = fread(configbuffer, 1, sizeof(configbuffer), file);
    
            memcpy(&MD5, configbuffer, 16);
            memcpy(&datasize, configbuffer+16, 4);

            int banknum = allocatedBankNum(datasize);
    //get outputbank list
            output_bankinfo = getOutputBankFromInstID(MD5);

            if(output_bankinfo == -1){
                printf("can't find output or output is brokenr\r\n");
                return;
            }
            output_startbank = (output_bankinfo >> 16) & 0x0000FFFF;
            output_endbank = output_bankinfo & 0x0000FFFF;

            buffer = (char *)malloc(datasize);
    //bank lock

            bankLock(output_startbank, output_endbank - output_startbank + 1);
        
            fclose(file);

    //data read from bank
            char filename[20];
            file = fopen(C2H_FILENAME, "wb");
            if(file == NULL){
                perror("Error opening file");
                return;
            }
            int cnt=0;
            for(int i=output_startbank; i<output_endbank+1; i++){    

                sprintf(filename, "%s%d", BANK_BASE_FILENAME, i);
                printf("readCallback : filename = %s\r\n", filename);
                mfile = fopen(filename, "rb");
                if(mfile == NULL){
                    perror("Error opening file for reading");
                    return;
                }
                if(datasize - cnt*BANKSIZE > BANKSIZE){
                    fread(buffer + cnt*BANKSIZE, sizeof(char), BANKSIZE, mfile);
                }else{
                    fread(buffer + cnt*BANKSIZE, sizeof(char), datasize - cnt*BANKSIZE, mfile);
                }
                cnt++;
                fclose(mfile);
            }
            

    //data write to c2h
            unsigned char dummy[64];
            fwrite(dummy, sizeof(char), sizeof(dummy), file);
            fwrite(buffer, sizeof(char), datasize, file);

    //bank unlock
            bankUnLock(output_startbank, output_endbank - output_startbank + 1);

    //C2H unlock
            shared_data->GBlockinfo &= ~C2H_CLEARMASK;
            shared_data->GBlockinfo &= ~C2H_MASK;

            free(buffer);
            fclose(file);

            bankRelease(MD5, 55);
            bankRelease(MD5, 56);
            bankRelease(MD5, 0);
            bankRelease(MD5, 1);
            bankRelease(MD5, 2);



            return;
        }
    }
    return;
}

void sigReleaseCallback(){
    MD5_t ID = shared_data->instID;

    int banks = getOutputBankFromInstID(ID);
    if(banks == -1){
        printf("can't find output or output is broken\r\n");
        return;
    }
    int startbank = (banks >> 16) & 0x0000FFFF;
    int outbank = banks & 0x0000FFFF;

    for(int i=0; i<BANKNUM; i++){
        shared_data->bankinfo[i].arg = -1;
    }
    return;
}



void QueueManager::instPush2Queue(int queuenum){

    memcpy(&(shared_data->instQueue[queuenum]), &(shared_data->requested_inst), sizeof(Inst));

    shared_data->requested_inst.optype = -1;
    order++;
    shared_data->instQueue[queuenum].order = order;
    return;
}

int QueueManager::queueCheck(int type){
    for(int i=0; i<8; i++){
        if(shared_data->instQueue[i].order == -type){
            return i;
        }
    }
    return -1;
}

int QueueManager::highOrderInst(){
    int high_order_inst = 2147483647;
    for(int i=0; i<8; i++){
        if(high_order_inst > shared_data->instQueue[i].order && 
                shared_data->instQueue[i].order != -1 &&
                shared_data->instQueue[i].order != -2){
            high_order_inst = shared_data->instQueue[i].order;
        }
    }
    if(high_order_inst == 2147483647)
        return -1;

    return high_order_inst;
}

void QueueManager::queueInstRelease(MD5_t instID, int order){
    for(int i=0; i<8; i++){
        if(MD5Compare(shared_data->instQueue[i].instID, instID) && 
                shared_data->instQueue[i].order == order){
            shared_data->instQueue[i].order = -1;
        }
    }

    return;
}
#endif