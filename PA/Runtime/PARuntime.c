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

//==============================required structures=========================================

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


//==============================runtime functions declaration=========================================

//for onnx-mlir accelerator runtime initialization
void OMInitAccelPA();
uint64_t OMInitCompatibleAccelPA(uint64_t versionNum);

//HW driver
int writeToAccel(float* data, int size, char* instID, int arg, int64_t* shape);
int pushInst(int optype, char* instID, int outsize, int config0, int config1, int config2,
             int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8, int chain);
void readFromAccel(float* data, int size, char* instID, int arg, int chain);
int waitInst(char* instID, int chain);

//simple runtime functions 
void runtimeConcat(float* input0, float* input1, float* out, int64_t* shape0, int64_t* shape1);
void runtimeSplit(float* input, float* out0, float* out1, int64_t* shape, 
                    int64_t split1, int64_t split2, int rank);
void runtimeResize(float* input, float* output, int64_t* shape);
void runtimeTranspose(float* input, float* output, int64_t* shape, int64_t* perm);
void runtimeSlice(float* input, int64_t* starts, int64_t* ends, int64_t* axes, int64_t* steps, float* output, 
                    int rank, int64_t* inputShape, int64_t* outputShape);

//utils
int64_t getLinearIndex(int64_t* shape, int64_t* indices, int rank);

//memory block
void h2cLock(SharedData* shm);
void h2cUnLock(SharedData* shm);
int h2cClearCheck(SharedData* shm);
void h2cSet(SharedData* shm);
void c2hLock(SharedData* shm);
void c2hUnLock(SharedData* shm);
int c2hClearCheck(SharedData* shm);

//for Hash
pid_t getPIDByName(const char *processName);
void computeMD5(const char* str, unsigned char* md5_result);




//==============================runtime functions define=========================================

int writeToAccel(float* data, int size, char* instID, int arg, int64_t* shape) {

#if _PRINT_
    printf("writeToAccel!\r\n");
#endif
    printf("<<writeToAccel>> : %s\r\n", instID);
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
    printf("------shape-------\r\n");
    printf("%d %d %d %d \r\n", shape[0], shape[1], shape[2], shape[3]);
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


void readFromAccel(float* data, int size, char* instID, int arg, int chain) {
#if _PRINT_
    printf("readFromAccel!\r\n");
#endif
    printf("<<readFromAccel>> : %s\r\n", instID);
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
    printf("readFromAccel >> filepointer0 : %ld\n", ftell(file));
    size_t bytesRead = fread(read_dummy, 1, sizeof(read_dummy), file); // read header
    printf("readFromAccel >> data pointer address: %p, filepointer: %ld, size: %d, datasize: %d, byteread: %zubytes\n",
                             data, ftell(file), size, datasize, bytesRead);
    fread(data, 1, datasize, file);


    printf("c2h file data : ");
    for(int i=0; i<20; i++){
        printf("%f ", *(data + i));
    }
    printf("\r\n");

    fclose(file);

    c2hUnLock(shared_data);
    shmdt(shared_data);

    return;
}

int pushInst(int optype, char* instID, int outsize, int config0, int config1, int config2,
             int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8, int chain) {

    printf("<<pushInst>>\r\n");
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT);
    SharedData *shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    Inst tempdata;

    tempdata.optype = optype;
    unsigned char MD5ID[16];
    computeMD5(instID, MD5ID);
    printf("pushInst >> instID : %s\r\n", instID);
    printf("pushInst >> MD5ID : ");
    for(int i=0; i<16; i++){
        printf("%02X", MD5ID[i]);
    }
    printf("\r\n");

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

int waitInst(char* instID, int chain) {
    printf("<<waitInst>>\r\n");
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

void runtimeSplit(float* input, float* out0, float* out1, int64_t* shape, 
                    int64_t split1, int64_t split2, int rank){
    printf("runtime Split >> Split!! \r\n");
    int64_t size = 1;

    for(int i=0; i<rank; i++){
        size *= shape[i];
    }
    printf("runtime Split >> size : %d\r\n", size);

    int64_t split_data_size = sizeof(float)*size / (split1 + split2);

    memcpy(out0, input, split_data_size * split1);
    memcpy(out1, input + split_data_size * split1 / 4, split_data_size * split2);

    return;
}

void runtimeConcat(float* input0, float* input1, float* out, int64_t* shape0, int64_t* shape1){
    int64_t size0 = 1;
    int64_t size1 = 1;

    for(int i=0; i<4; i++){
        size0 *= shape0[i];
        size1 *= shape1[i];
    }

    int64_t data_size0 = sizeof(float)*size0;
    int64_t data_size1 = sizeof(float)*size1;

    memcpy(out, input0, data_size0);
    memcpy(out + data_size0/4, input1, data_size1);

    return;
}

void runtimeResize(float* input, float* output, int64_t* shape){
    
    int N = shape[0];   // Batch size
    int C = shape[1];   // Number of channels
    int H = shape[2];   // Input height
    int W = shape[3];   // Input width

    int outH = H * 2;
    int outW = W * 2;

    // N, C, H, W 순서로 입력을 순회하면서 nearest 방식으로 값을 복사
    for (int n = 0; n < N; ++n) {          // Batch loop
        for (int c = 0; c < C; ++c) {      // Channel loop
            for (int h = 0; h < outH; ++h) {  // Output height loop
                for (int w = 0; w < outW; ++w) {  // Output width loop
                    // Nearest neighbor 계산: floor(h / 2), floor(w / 2)
                    int inH = (int)floor(h / 2.0);  // 입력 높이 매핑
                    int inW = (int)floor(w / 2.0);  // 입력 너비 매핑

                    // 입력과 출력의 인덱스를 계산합니다.
                    int inputIndex = ((n * C + c) * H + inH) * W + inW;  // 입력 인덱스
                    int outputIndex = ((n * C + c) * outH + h) * outW + w;  // 출력 인덱스

                    // Nearest neighbor 방식으로 값을 복사
                    output[outputIndex] = input[inputIndex];
                }
            }
        }
    }
    return;
}

void runtimeTranspose(float* input, float* output, int64_t* shape, int64_t* perm){
    // Shape information for NCHW
    int N = shape[0];
    int C = shape[1];
    int H = shape[2];
    int W = shape[3];

    // Output shape after permutation
    int64_t outShape[4];
    for (int i = 0; i < 4; i++) {
        outShape[i] = shape[perm[i]];
    }

    // Iterate over all elements in the input tensor and permute them
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    // Original index (NCHW format)
                    int inputIndex = ((n * C + c) * H + h) * W + w;

                    // Calculate output index based on the permutation
                    int outputIndex = 0;
                    int coords[4] = {n, c, h, w};  // Original NCHW coordinates
                    int strides[4] = {outShape[1] * outShape[2] * outShape[3],
                                      outShape[2] * outShape[3],
                                      outShape[3],
                                      1};
                    for (int i = 0; i < 4; i++) {
                        outputIndex += coords[perm[i]] * strides[i];
                    }

                    // Perform the transpose by copying the value to the permuted location
                    output[outputIndex] = input[inputIndex];
                }
            }
        }
    }
}

void runtimeSlice(float* input, int64_t* starts, int64_t* ends, int64_t* axes, int64_t* steps, float* output, 
                    int rank, int64_t* inputShape, int64_t* outputShape){
    // Allocate memory for input and output indices
    int64_t* inputIndices = (int64_t*)malloc(rank * sizeof(int64_t));
    int64_t* outputIndices = (int64_t*)malloc(rank * sizeof(int64_t));
    
    // Initialize input indices with start values
    for (int i = 0; i < rank; ++i) {
        inputIndices[i] = starts[i];
        outputIndices[i] = 0;  // Output indices always start from 0
    }

    int done = 0;  // Flag to check if slicing is done
    while (!done) {
        // Calculate the linear index for input and output
        int64_t inputIndex = getLinearIndex(inputShape, inputIndices, rank);
        int64_t outputIndex = getLinearIndex(outputShape, outputIndices, rank);
        
        // Copy the value from input to output
        output[outputIndex] = input[inputIndex];

        // Update indices
        done = 1;  // Assume done unless further updates
        for (int axis = 0; axis < rank; ++axis) {
            inputIndices[axis] += steps[axis];
            outputIndices[axis]++;
            
            if (inputIndices[axis] < ends[axis]) {
                done = 0;  // Not done, continue
                break;
            } else if (axis < rank - 1) {
                inputIndices[axis] = starts[axis];  // Reset current axis
                outputIndices[axis] = 0;
            }
        }
    }

    // Free allocated memory
    free(inputIndices);
    free(outputIndices);

}

void OMInitAccelPA() {
    printf("OMInitAccelPA!\r\n");
    return;
}

uint64_t OMInitCompatibleAccelPA(uint64_t versionNum){
    printf("Compatible PA Accelerator Init!!\r\n");
    return -1;
}

int64_t getLinearIndex(int64_t* shape, int64_t* indices, int rank) {
    int64_t index = 0;
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
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