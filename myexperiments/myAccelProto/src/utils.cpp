#ifndef UTILS_CPP
#define UTILS_CPP

#include <iostream>
#include <stdint.h>
#include <string.h>
#include "utils.h"
#include "controller.h"

extern SharedData *shared_data;

int seekBank(int size){
    int requiredBanks = size/BANKSIZE + bool(size%BANKSIZE);
    bool canAllocate;
    int i, j;
    for(i=0; i<=BANKNUM - requiredBanks; i++){
        canAllocate = true;

        for(j=0; j<requiredBanks; j++){
            if(shared_data->bankinfo[i+j].arg != -1){
                canAllocate = false;
                break;
            }
        }

        if(canAllocate){
            return i;
        }
    }
    return -1;
}

int allocatedBankNum(int size){
    return size/BANKSIZE + bool(size%BANKSIZE);
}

int getOutputBankFromInstID(MD5_t ID){
    int start = -1;
    int end = -1;
    for(int i=0; i<BANKNUM; i++){
        if(MD5Compare(shared_data->bankinfo[i].instID, ID)){
            if(shared_data->bankinfo[i].arg == 56){
                start = i;
                end = i;
            }

            if(shared_data->bankinfo[i].arg == 55){
                if(i-end != 1)
                    return -1;
                end = i;
            }
        }
    }

    return (start<<16 | end);
}

int getBank(MD5_t ID, int arg){
    int start = -1;
    int end = -1;
    for(int i=0; i<BANKNUM; i++){
        if(MD5Compare(shared_data->bankinfo[i].instID, ID)){
            if(shared_data->bankinfo[i].arg == arg){
                if (start == -1)
                    start = i;
                else
                    if(i-end != 1){return -1;}
                end = i;
            }
        }
    }
    return (start<<16 | end);
}

bool MD5Compare(MD5_t ID1, MD5_t ID2){

    int id1[4];
    int id2[4];

    memcpy(id1, &ID1, 16);
    memcpy(id2, &ID2, 16);

    for(int i=0; i<4; i++){
        if(id1[i] != id2[i])
            return false;
    }
    return true;
}

void bankLock(int startbank, int banknum){
    for(int i=0; i<banknum; i++){
        int bankIndex = startbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
        
        if(!(shared_data->banklockinfo[arrayIndex] & (1 << bitPosition)))
            shared_data->banklockinfo[arrayIndex] |= (1 << bitPosition);
        else
            i--;
    }    
}

void bankUnLock(int startbank, int banknum){
    for(int i=0; i<banknum; i++){
        int bankIndex = startbank + i;
        int arrayIndex = bankIndex / 32;  
        int bitPosition = bankIndex % 32; 
        
        shared_data->banklockinfo[arrayIndex] &= (0 << bitPosition);
    }
    return;
}

void bankRelease(MD5_t ID, int arg){
    for(int i=0; i<BANKNUM; i++){
        if(MD5Compare(shared_data->bankinfo[i].instID, ID) 
                && (shared_data->bankinfo[i].arg == arg)){
            shared_data->bankinfo[i].arg = -1;
        }
    }
}

#endif