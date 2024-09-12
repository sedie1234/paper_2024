#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <iostream>

#define BANKSIZE        0x10000
#define BANKNUM         300

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
    int8_t GBlockinfo; //0b000000XY // X:h2c // Y:c2h //0:unlock, 1:lock
    int banklockinfo[BANKNUM/32 + bool(BANKNUM%32)]; //192bit banklock info //0:unlock, 1:lock
} SharedData;

SharedData *shared_data;

void dumpInst(Inst data);
void dumpSharedMemory();
void dumpBank();
void dumpQueue();
void printCMD();

#include <iostream>
#include <string>

int main() {
    std::string input;

    printf("shared mem size : %d \r\n", sizeof(SharedData));
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666|IPC_CREAT);
    shared_data = (SharedData*) shmat(shmid, (void*)0, 0);

    if (shmid < 0) {
        perror("shmget");
        exit(1);
    }

    while (true) {
        
        std::getline(std::cin, input);  // 외부 입력을 받습니다.

        if (input == "q") {
            std::cout << "Exiting program..." << std::endl;
            shmdt(shared_data);
            shmctl(shmid, IPC_RMID, NULL);
            break;  // 루프를 종료하고 프로그램을 끝냅니다.
        }else if(input == "r"){
            dumpSharedMemory();
        }else if(input == "b"){
            dumpBank();
        }else if(input == "g"){
            dumpQueue();
        }else if(input == "h"){
            printCMD();
        }


        std::cout << "You entered: " << input << std::endl;
    }

    return 0;
}

void dumpQueue(){
    for(int i=0; i<8; i++){
        printf("\r\n[instQueue %d] %d | %d | ", i, shared_data->instQueue[i].order, shared_data->instQueue[i].optype);
            
        for (int j = 0; j < 16; j++) {
            printf("%02x", shared_data->instQueue[i].instID.bytes[j]);
        }
        printf("\r\n\t%d", shared_data->instQueue[i].outsize);
        for (int j=0; j<3; j++){
            printf(" %d", shared_data->instQueue[i].config[j]);
        }
        for (int j=0; j<9; j++){
            printf(" %d", shared_data->instQueue[i].arg[j]);
        }
    }
    printf("\r\n");
    
}

void printCMD(){
    printf("=====CMD=====\r\n");
    printf("r : shared memory dump\r\n");
    printf("b : bank dump\r\n");
    printf("g : inst Queue dump\r\n");
    printf("q : quit\r\n");
    printf("h : help\r\n");
}

void dumpInst(Inst data){
    printf("order | optype : %d %d\n", data.order, data.optype);

    printf("MD5 : ");
    for (int i = 0; i < 16; i++) {
        printf("%02x", data.instID.bytes[i]);  // MD5 출력 시 16진수 형식으로 출력
    }
    printf("\n");

    printf("size : %d\n", data.outsize);

    printf("configs & args\n");

    for (int i = 0; i < 3; i++) {
        printf("%d ", data.config[i]);
    }
    printf("\n");

    for (int i = 0; i < 9; i++) {
        printf("%d ", data.arg[i]);
    }
    printf("\n");

}

void dumpSharedMemory() {

    printf("interrupt class :     %d\n", shared_data->interrupt_class);
    printf("interrupt class :     %d\n", shared_data->interrupt_class);

    dumpInst(shared_data->requested_inst);
    printf("%02X\r\n", shared_data->GBlockinfo);
    for(auto bank : shared_data->banklockinfo){
        printf("%08X\r\n", bank);
    }

}

void dumpBank(){
    for(int i=0; i<192; i++){
        printf("[bank%03d info] ", i);
        for(int j=0; j<16; j++){
            printf("%02X", shared_data->bankinfo[i].instID.bytes[j]);
        }
        printf(" %d ", shared_data->bankinfo[i].arg);
        for(int j=0; j<4; j++){
            printf("%d ", shared_data->bankinfo[i].shape[j]);
        }
        printf("\r\n");
    }
}