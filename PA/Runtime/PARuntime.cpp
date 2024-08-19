#pragma once

#include <pthread.h>


#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <signal.h>
#include <openssl/md5.h>
#include <vector>
#include <string>
#include <iomanip>
#include <string.h>

#define ACCEL_PROCESS       "MyAccel"

#define H2C_FILENAME        "./device/h2c"
#define C2H_FILENAME        "./device/c2h"

#define H2C_MASK            0x01
#define C2H_MASK            0x02

#define BANKSIZE        0x10000
#define BANKNUM         192

#define TIMEDELAY       1000 //us

enum{
    SIG_WRITE,
    SIG_READ,
    SIG_RELEASE
};


typedef struct {
    unsigned char bytes[16];
} MD5_t;

typedef struct {
    int order;
    int8_t optype;
    MD5_t instID;
    int outsize;
    int8_t config[3];
    int arg[9];
} Inst;

typedef struct {
    //int8_t index;
    MD5_t instID;
    int8_t arg;
    int16_t shape[4];
} BankInfo_t;

typedef struct {
    int8_t interrupt_class;
    Inst requested_inst;
    Inst instQueue[8];
    BankInfo_t bankinfo[BANKNUM];
    MD5_t instID;
    int8_t GBlockinfo; //000000XY // X:h2c // Y:c2h //0:unlock, 1:lock
    int banklockinfo[BANKNUM/32 + bool(BANKNUM%32)]; //192bit banklock info //0:unlock, 1:lock
} SharedData;


void h2cLock(SharedData* shm);
void h2cUnLock(SharedData* shm);
void c2hLock(SharedData* shm);
void c2hUnLock(SharedData* shm);
pid_t getPIDByName(const char *processName);
std::array<unsigned char, 16> stringToMD5(const std::string &str);

void writeToAccel(float* data, int size, std::string instID, int arg, int64_t* shape);
void readFromAccel(float* data, int size, std::string instID, int arg);
void pushInst(int optype, std::string instID, int outsize, int config0, int config1, int config2,
            int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8);
void waitInst(std::string instID);



void writeToAccel(float* data, int size, std::string instID, int arg, int64_t* shape){

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666|IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    h2cLock(shared_data);

    pid_t pid = getPIDByName(ACCEL_PROCESS);

    if (pid == -1) {
        printf("Failed to get PID for process %s\n", ACCEL_PROCESS);
    } else {
        printf("PID of %s: %d\n", ACCEL_PROCESS, pid);
    }

//h2c file write
    FILE* file;

    file = fopen(H2C_FILENAME, "wb");
    if(file == nullptr){
        printf("Failed to open file for writing\r\n");
    }

    std::array<unsigned char, 16> MD5ID = stringToMD5(instID);
    char header_dummy[3];
    int16_t _shape[4];
    for(int i=0; i<4; i++){
        _shape[i] = (*(shape + i))&0x0000FFFF; //overflow 가능성
    }

    int datasize = size*sizeof(float);

    fwrite(MD5ID.data(), 1, MD5ID.size(), file);        //header : MD5 16byte
    fwrite(&datasize, 1, 4, file);                          //header : size 4byte
    fwrite(&arg, 1, 1, file);                           //header : usage 1byte
    fwrite(_shape, 1, 8, file);                         //header : shape info 8byte
    fwrite(header_dummy, 1, sizeof(header_dummy), file);//header : dummy for 32byte align
    
    fwrite(data, 1, datasize, file);          //data

    fclose(file);

//send interrupt
    shared_data->interrupt_class = SIG_WRITE;

    if(kill(pid, SIGUSR1) == -1){
        perror("Error sending signal");
		exit(EXIT_FAILURE);
    }

    h2cUnLock(shared_data);
    shmdt(shared_data);

    return;
}


void readFromAccel(float* data, int size, std::string instID, int arg){
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666|IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);


    c2hLock(shared_data);

    pid_t pid = getPIDByName(ACCEL_PROCESS);

    if (pid == -1) {
        printf("Failed to get PID for process %s\n", ACCEL_PROCESS);
    } else {
        printf("PID of %s: %d\n", ACCEL_PROCESS, pid);
    }

//c2h file write
    FILE* file;

    file = fopen(C2H_FILENAME, "wb");
    if(file == nullptr){
        printf("Failed to open file for writing\r\n");
    }

    std::array<unsigned char, 16> MD5ID = stringToMD5(instID);

    int datasize = size*sizeof(float);
    unsigned char dummy[12];

    fwrite(MD5ID.data(), 1, MD5ID.size(), file);        //header : MD5 16byte
    fwrite(&datasize, 1, 4, file);
    fwrite(dummy, 1, sizeof(dummy), file);

    fclose(file);
    c2hUnLock(shared_data);

//send interrupt
    shared_data->interrupt_class = SIG_READ;

    if(kill(pid, SIGUSR1) == -1){
        perror("Error sending signal");
		exit(EXIT_FAILURE);
    }

//wait a minute
    usleep(TIMEDELAY);

//c2h file read
    c2hLock(shared_data);

    file = fopen(C2H_FILENAME, "rb");
    if(file == nullptr){
        printf("Failed to open file for writing\r\n");
    }

    unsigned char read_dummy[32];
    fread(read_dummy, 1, 32, file);             //read header
    fread(data, 1, datasize, file);

    fclose(file);

    c2hUnLock(shared_data);
    shmdt(shared_data);

    return;
}

void pushInst(int optype, std::string instID, int outsize, int config0, int config1, int config2,
            int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8){
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666|IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    Inst tempdata;

    tempdata.optype = optype;
    std::array<unsigned char, 16> MD5ID = stringToMD5(instID);
    memcpy(&(tempdata.instID), MD5ID.data(), sizeof(MD5ID));
    tempdata.outsize = outsize;
    tempdata.config[0] = config0;
    tempdata.config[1] = config1;
    tempdata.config[2] = config2;
    tempdata.arg[0] = arg0;
    tempdata.arg[1] = arg1;
    tempdata.arg[2] = arg2;
    tempdata.arg[3] = arg3;
    tempdata.arg[4] = arg4;
    tempdata.arg[5] = arg5;
    tempdata.arg[6] = arg6;
    tempdata.arg[7] = arg7;
    tempdata.arg[8] = arg8;
    
    while(true){
        if(shared_data->requested_inst.optype == -1){
            memcpy(&(shared_data->requested_inst), &tempdata, sizeof(Inst));

            shmdt(shared_data);
            return;    
        }
    }
    
    return;

}

void waitInst(std::string instID){
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666|IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    std::array<unsigned char, 16> MD5ID = stringToMD5(instID);

    while(true){

        int i;
        int flag;
        for(i=0; i<8; i++){
            flag = 0;
            for(int j=0; j<16; j++){
                if(MD5ID[j] != shared_data->instQueue[i].instID.bytes[j]){
                    flag = 1;
                    break;
                }
            }
            if(flag == 0)
                break;
        }

        if(flag == 0){
            if(shared_data->instQueue[i].order == -2){
                shared_data->instQueue[i].order = -1;
                return;
            }
        }
            
    }

    return;
}

void OMInitAccelPA() {
    printf("OMInitAccelPA!\r\n");
    return;
}


uint64_t OMInitCompatibleAccelPA(uint64_t versionNum){
    // printf("Compatible PA Accelerator Init!!\r\n");
    return -1;
}


void h2cLock(SharedData* shm){
    while(true){
        if((shm->GBlockinfo)&H2C_MASK == 0){
            (shm->GBlockinfo) |= H2C_MASK;
            return;
        }
    }
    
}

void h2cUnLock(SharedData* shm){
    (shm->GBlockinfo) &= ~H2C_MASK;
    return;
}

void c2hLock(SharedData* shm){
    while(true){
        if((shm->GBlockinfo)&C2H_MASK == 0){
            (shm->GBlockinfo) |= C2H_MASK;
            return;
        }
    }
}

void c2hUnLock(SharedData* shm){
    (shm->GBlockinfo) &= ~C2H_MASK;
    return;
}

pid_t getPIDByName(const char *processName) {
    char command[256];
    snprintf(command, sizeof(command), "pidof %s", processName);
    
    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        perror("popen failed");
        return -1;
    }

    pid_t pid;
    if (fscanf(fp, "%d", &pid) != 1) {
        pclose(fp);
        return -1;
    }

    pclose(fp);
    return pid;
}

std::array<unsigned char, 16> stringToMD5(const std::string &str){

    std::array<unsigned char, MD5_DIGEST_LENGTH> result;

    MD5(reinterpret_cast<const unsigned char*>(str.c_str()), str.length(), result.data());

    return result;
}