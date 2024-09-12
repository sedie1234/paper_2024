#include <pthread.h>


#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <signal.h>
#include <openssl/md5.h>
#include <string.h>

#define _PRINT_             0

#define ACCEL_PROCESS       "myaccel"

#define H2C_FILENAME        "../myAccelProto/run/device/h2c"
#define C2H_FILENAME        "../myAccelProto/run/device/c2h"

#define H2C_MASK            0x01
#define H2C_CLEARMASK       0x02
#define C2H_MASK            0x04
#define C2H_CLEARMASK       0x08

#define BANKSIZE        0x10000
#define BANKNUM         300


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
    int64_t shape[4];
} BankInfo_t;

typedef struct {
    int8_t interrupt_class;
    Inst requested_inst;
    Inst instQueue[8];
    BankInfo_t bankinfo[BANKNUM];
    MD5_t instID;
    int8_t GBlockinfo; //000000XY // X:h2c // Y:c2h //0:unlock, 1:lock
    int banklockinfo[BANKNUM/32 + (BANKNUM % 32 != 0)];
} SharedData;

int writeToAccel(float* data, int size, char* instID, int arg, int64_t* shape);
int pushInst(int optype, char* instID, int outsize, int config0, int config1, int config2,
             int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8, int chain);
void readFromAccel(float* data, int size, char* instID, int arg, int chain);

void h2cLock(SharedData* shm);
void h2cUnLock(SharedData* shm);
int h2cClearCheck(SharedData* shm);
void h2cSet(SharedData* shm);

void c2hLock(SharedData* shm);
void c2hUnLock(SharedData* shm);
int c2hClearCheck(SharedData* shm);
pid_t getPIDByName(const char *processName);
void computeMD5(const char* str, unsigned char* md5_result);

//shape instID 

void sigTest(int64_t* shape, char* instID);
void mulTest(int64_t* shape, char* instID);
void convTest(int64_t* shape, char* instID, int channel);



// ./runtime [optype] [instID] [shape[0]] [shape[1]] [shape[2]] [shape[3]] [channel]

int main(int argc, char** argv){

    float data[500000];
    float recv_data[500000];
    int64_t optype=atoi(argv[1]);
    char* instID = argv[2];
    int64_t shape[4] = {atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6])};
    int channel = atoi(argv[7]);

    if(optype == 0){                            //TBD 

    }else if(optype == 1){                      //conv
        convTest(shape, instID, channel);
    }else if(optype == 2){                      //sigmoid
        sigTest(shape, instID);
    }else if(optype == 3){                      //mul
        mulTest(shape, instID);
    }


    return 0;
}

void convTest(int64_t* shape, char* instID, int channel){

    int padding = 1;
    int strides = 2;


    printf("======[Runtime Test : Conv]======\r\n");

    int64_t inputsize = 1;
    for(int i=0; i<4; i++){
        inputsize *= shape[i];
    }

    int64_t kernel_shape[4] = {channel, shape[1], 3, 3};
    int64_t bias_shape[4] = {1, 1, 1, channel};

    int64_t kernelsize = 1;
    int64_t biassize = 1;

    for(int i=0; i<4; i++){
        kernelsize *= kernel_shape[i];
    }

    for(int i=0; i<4; i++){
        biassize *= bias_shape[i];
    }

    float* input = (float*)malloc(sizeof(float)*inputsize);
    float* kernel = (float*)malloc(sizeof(float)*kernelsize);
    float* bias = (float*)malloc(sizeof(float)*biassize);

    int index = 0;

    for (int h = 0; h < 5; h++) {
        for (int w = 0; w < 5; w++) {
            input[h * shape[3] + w] = (float)(index + 1);  // 예: 1.0f, 2.0f, ..., 25.0f
            index++;
        }
    }

    // // 나머지 값은 1.0f로 초기화
    // for (int i = 25; i < inputsize; i++) {
    //     input[i] = 1.0f;
    // }


    for (int i = 0; i < 9; i++) {
        kernel[i] = 0.1f * (i + 1);  // 예: 0.1f, 0.2f, ..., 0.9f
    }
    // // 나머지 값은 1.0f로 초기화
    // for (int i = 9; i < kernelsize; i++) {
    //     kernel[i] = 1.0f;
    // }

    for (int i = 0; i < biassize; i++) {
        bias[i] = 0.1f * i;  // 예: 0.0f, 0.1f, 0.2f, ..., 3.1f
    }


    printf("runtime >> writeToAccel start! : input\r\n");
    int a = writeToAccel(input, inputsize, instID, 0, shape);
    
    printf("runtime >> writeToAccel start! : kernel\r\n");
    int b = writeToAccel(kernel, kernelsize, instID, 1, kernel_shape);

    printf("runtime >> writeToAccel start! : bias\r\n");
    int c = writeToAccel(bias, biassize, instID, 2, bias_shape);


    int outH = (shape[2] + 2*padding - kernel_shape[2]) / strides + 1;
    int outW = (shape[3] + 2*padding - kernel_shape[3]) / strides + 1;

    int64_t output_shape[4] = {1, kernel_shape[0], outH, outW};
    int64_t size = 1;

    for(int i=0; i<4; i++){
        size *= output_shape[i];
    }

    float* recv_data = (float*)malloc(sizeof(float)*size);

    printf("runtime >> pushInst start!\r\n");
    int d = pushInst(1, instID, size, kernel_shape[2], padding, strides, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0);

    printf("runtime >> waitInst start!\r\n");
    if(waitInst(instID, 0) == -1)
        printf("wait fault!\r\n");
    printf("runtime >> readFromAccel start!\r\n");
    readFromAccel(recv_data, size, instID, 56, 0);

    printf("-----result-----\r\n");
    printf("--------input-------\r\n");
    for (int h = 0; h < 5; h++) {
        for (int w = 0; w < 5; w++) {
            printf("%f ",input[h * shape[3] + w]);
        }
        printf("\r\n");
    }
    printf("--------kernel-------\r\n");
    for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 3; w++) {
            printf("%f ",kernel[h * kernel_shape[3] + w]);
        }
        printf("\r\n");
    }
    printf("--------output-------\r\n");
    printf("output 0 : %f\r\n", recv_data[0]);
    printf("output 1 : %f\r\n", recv_data[1]);

    printf("...\r\n");
    printf("======[Test End : Conv]======\r\n");

    free(input);
    free(kernel);
    free(bias);
    free(recv_data);

}

void mulTest(int64_t* shape, char* instID){

    printf("======[Runtime Test : Mul]======\r\n");

    int64_t size = 1;
    for(int i=0; i<4; i++){
        size *= shape[i];
    }

    float* data0 = (float*)malloc(sizeof(float)*size);
    float* data1 = (float*)malloc(sizeof(float)*size);
    float* recv_data = (float*)malloc(sizeof(float)*size);

    data0[0] = 0.1;
    data0[1] = 0.5;
    data0[2] = 1.1;
    data0[3] = 2.5;
    data0[4] = 3.3;
    data0[5] = 4.4;
    data0[6] = 5.5;
    data0[7] = 6.75;
    data0[8] = 7.125;
    data0[9] = 8.2;
    data0[10] = 9.9;
    data0[11] = 10.01;
    data0[12] = 11.11;
    data0[13] = 12.25;
    data0[14] = 13.333;
    data0[15] = 14.5;
    data0[16] = 15.75;
    data0[17] = 16.875;
    data0[18] = 19.99;
    data0[19] = 0.2;
    
    data1[0] = 0.2;
    data1[1] = 0.4;
    data1[2] = 2.0;
    data1[3] = 3.0;
    data1[4] = 3.3;
    data1[5] = 2.5;
    data1[6] = 1.1;
    data1[7] = 4.0;
    data1[8] = 2.0;
    data1[9] = 5.0;
    data1[10] = 0.5;
    data1[11] = 1.1;
    data1[12] = 9.0;
    data1[13] = 0.8;
    data1[14] = 3.0;
    data1[15] = 2.0;
    data1[16] = 4.0;
    data1[17] = 0.4;
    data1[18] = 1.01;
    data1[19] = 5.5;

    printf("runtime >> writeToAccel start! : data0\r\n");
    int a = writeToAccel(data0, size, instID, 0, shape);
    
    printf("runtime >> writeToAccel start! : data1\r\n");
    int b = writeToAccel(data1, size, instID, 1, shape);

    printf("runtime >> pushInst start!\r\n");
    int c = pushInst(3, instID, size, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    printf("runtime >> waitInst start!\r\n");
    if(waitInst(instID, 0) == -1)
        printf("wait fault!\r\n");
    printf("runtime >> readFromAccel start!\r\n");
    readFromAccel(recv_data, size, instID, 56, 0);

    printf("-----result-----\r\n");
    printf("    A   x     B   =     C\r\n");
    for(int i=0; i<20; i++){
        printf("%05f x %05f = %05f\r\n", data0[i], data1[i], recv_data[i]);
    }

    printf("...\r\n");
    printf("======[Test End : Mul]======\r\n");

    free(data0);
    free(data1);
    free(recv_data);


}

void sigTest(int64_t* shape, char* instID){

    printf("======[Runtime Test : Sigmoid]======\r\n");

    int64_t size = 1;
    for(int i=0; i<4; i++){
        size *= shape[i];
    }

    float* data = (float*)malloc(sizeof(float)*size);
    float* recv_data = (float*)malloc(sizeof(float)*size);

    data[0] = -10;
    data[1] = -5;
    data[2] = -2;
    data[3] = -1;
    data[4] = -0.5;
    data[5] = 0;
    data[6] = 0.5;
    data[7] = 1;
    data[8] = 2;
    data[9] = 5;


    printf("runtime >> writeToAccel start!\r\n");
    int a = writeToAccel(data, size, instID, 0, shape);
    printf("runtime >> pushInst start!\r\n");
    int b = pushInst(2, instID, size, 1, 2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    printf("runtime >> waitInst start!\r\n");
    if(waitInst(instID, 0) == -1)
        printf("wait fault!\r\n");
    printf("runtime >> readFromAccel start!\r\n");
    readFromAccel(recv_data, size, instID, 56, 0);

    printf("recv_data :");
    for(int i=0; i<10; i++){
        printf(" %f", recv_data[i]);
    }
    printf("...\r\n");
    printf("======[Test End : Sigmoid]======\r\n");

    free(data);
    free(recv_data);

    return;

}

int writeToAccel(float* data, int size, char* instID, int arg, int64_t* shape) {

#if _PRINT_
    printf("writeToAccel!\r\n");
#endif

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT);
    SharedData *shared_data = (SharedData*)shmat(shmid, (void*)0, 0);

    while(h2cClearCheck(shared_data)==0);

    h2cLock(shared_data);

    printf("pid name : %s \r\n", ACCEL_PROCESS);

    pid_t pid = getPIDByName(ACCEL_PROCESS);

    if (pid == -1) {
        printf("Failed to get PID for process %s\n", ACCEL_PROCESS);
    } else {
        printf("PID of %s: %d\n", ACCEL_PROCESS, pid);
    }

    // h2c file write
    FILE* file;

    file = fopen(H2C_FILENAME, "wb");
    if (file == NULL) {
        printf("Failed to open file for writing\r\n");
        return -1;
    }
#if _PRINT_
    printf("%s file open\r\n", H2C_FILENAME);
#endif
    printf("inst ID : %s \r\n", instID);
    unsigned char MD5ID[16];
    computeMD5(instID, MD5ID);
#if _PRINT_
    printf("get MD5\r\n");
#endif

    char header_dummy[11] = {0};

    int datasize = size * sizeof(float);
    printf("write md5 to h2c file : ");
    for(int i=0; i<16; i++){
        printf("%02X", MD5ID[i]);
    }
    printf("\r\n");

    printf(" datasize : %d \r\n", datasize);

    int8_t arg8bit = arg&0xFF;
    printf("write arg : %d\r\n", arg8bit);
    fwrite(MD5ID, 1, sizeof(MD5ID), file);            // header : MD5 16byte
    fwrite(&datasize, 1, 4, file);                    // header : size 4byte
    fwrite(&arg8bit, 1, 1, file);                         // header : usage 1byte
    fwrite(shape, 1, sizeof(shape)*4, file);          // header : shape info 32byte
    fwrite(header_dummy, 1, sizeof(header_dummy), file); // header : dummy for 11byte align
    
    fwrite(data, 4, size, file);                  // data

    printf("h2c file data : ");
    for(int i=0; i<20; i++){
        printf("%f ", *(data + i));
    }
    printf("\r\n");

    fclose(file);

    // send interrupt
    shared_data->interrupt_class = SIG_WRITE;

    if (kill(pid, SIGUSR1) == -1) {
        perror("Error sending signal");
        exit(EXIT_FAILURE);
    }

    h2cSet(shared_data);
    h2cUnLock(shared_data);
    shmdt(shared_data);

    return 0;
}

int waitInst(char* instID, int chain) {
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    unsigned char MD5ID[16];
    computeMD5(instID, MD5ID);

    while (1) {
        int flag = 0;
        int i, j;

        for (i = 0; i < 8; i++) {
            flag = 0;
            for (j = 0; j < 16; j++) {
                if (MD5ID[j] != shared_data->instQueue[i].instID.bytes[j]) {
                    flag = 1;
                    break;
                }
            }

            if (flag == 0) {
                if(shared_data->instQueue[i].order == -2){
                    shared_data->instQueue[i].order = -1;
                    shmdt(shared_data);
                    return 0;
                }
            }
        }

    }

    shmdt(shared_data);
    return -1;
}

void readFromAccel(float* data, int size, char* instID, int arg, int chain) {
#if _PRINT_
    printf("readFromAccel!\r\n");
#endif
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    c2hLock(shared_data);

    pid_t pid = getPIDByName(ACCEL_PROCESS);

    if (pid == -1) {
        printf("Failed to get PID for process %s\n", ACCEL_PROCESS);
    } else {
        printf("PID of %s: %d\n", ACCEL_PROCESS, pid);
    }

    // c2h file write
    FILE* file;
    file = fopen(C2H_FILENAME, "wb");
    if (file == NULL) {
        printf("Failed to open file for writing\r\n");
    }

    unsigned char MD5ID[16];
    computeMD5(instID, MD5ID);

    int datasize = size * sizeof(float);
    unsigned char dummy[12] = {0};

    fwrite(MD5ID, 1, sizeof(MD5ID), file);   // header : MD5 16 byte
    fwrite(&datasize, 1, 4, file);
    fwrite(dummy, 1, sizeof(dummy), file);

    fclose(file);


    c2hSet(shared_data);
    c2hUnLock(shared_data);

    // send interrupt
    shared_data->interrupt_class = SIG_READ;

    if (kill(pid, SIGUSR1) == -1) {
        perror("Error sending signal");
        exit(EXIT_FAILURE);
    }

    // c2h file read
    while(c2hClearCheck(shared_data) == 0);
    c2hLock(shared_data);

    file = fopen(C2H_FILENAME, "rb");
    if (file == NULL) {
        printf("Failed to open file for reading\r\n");
    }

    unsigned char read_dummy[64];
    fread(read_dummy, 1, sizeof(read_dummy), file); // read header
    fread(data, 1, datasize, file);

    fclose(file);

    c2hUnLock(shared_data);
    shmdt(shared_data);

    return;
}

int pushInst(int optype, char* instID, int outsize, int config0, int config1, int config2,
             int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8, int chain) {
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    Inst tempdata;

    tempdata.optype = optype;
    unsigned char MD5ID[16];
    computeMD5(instID, MD5ID);
    memcpy(&(tempdata.instID), MD5ID, sizeof(MD5ID));
    tempdata.outsize = outsize * sizeof(float);
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
    
    while (1) {
        if (shared_data->requested_inst.optype == -1) {
            memcpy(&(shared_data->requested_inst), &tempdata, sizeof(Inst));

            shmdt(shared_data);
            return 0;    
        }
    }
    
    return -1;
}

void h2cLock(SharedData* shm){
    printf("h2c lock : wait available time to use the h2c file \r\n");
    while(1){
        if(((shm->GBlockinfo)&H2C_MASK) == 0){
            (shm->GBlockinfo) |= H2C_MASK;
            printf("h2clock\r\n");
            return;
        }
    }
    
}

int h2cClearCheck(SharedData* shm){
    if(((shm->GBlockinfo)&H2C_CLEARMASK) == 0)
        return 1;
    else
        return 0;

}

void h2cUnLock(SharedData* shm){
    (shm->GBlockinfo) &= ~H2C_MASK;
    return;
}

void h2cSet(SharedData* shm){
    (shm->GBlockinfo) |= H2C_CLEARMASK;
}

void c2hLock(SharedData* shm){
    while(1){
        if(((shm->GBlockinfo)&C2H_MASK) == 0){
            (shm->GBlockinfo) |= C2H_MASK;
            return;
        }
    }
}

int c2hClearCheck(SharedData* shm){
    if(((shm->GBlockinfo)&C2H_CLEARMASK) == 0)
        return 1;
    else
        return 0;
}

void c2hSet(SharedData* shm){
    (shm->GBlockinfo) |= C2H_CLEARMASK;
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

void computeMD5(const char* str, unsigned char* md5_result) {
    // 문자열의 길이를 계산합니다.
    printf("compute MD5\r\n");
    size_t length = strlen(str);

    printf("hash length : %d\r\n", length);
    // MD5 함수를 사용하여 해시를 계산하고 결과를 result 배열에 저장합니다.
    MD5((const unsigned char*)str, length, md5_result);
}

