#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <iostream>

#define _PRINT_         1

#define H2C_FILENAME        "./device/h2c"
#define C2H_FILENAME        "./device/c2h"
#define BANK_BASE_FILENAME  "./memory/"

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

#include "utils.h"

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


//class
class QueueManager {
    
public:
    QueueManager(){
        order = 0;

    };

    int queueCheck(int type); // type=1:blank? // type=2:complete?
    void instPush2Queue(int queuenum);
    int highOrderInst();
    void queueInstRelease(MD5_t instID, int order);

    int order;

};

//functions
void signal_handler(int signum);
void sharedDataInit();

void sigWriteCallback();
void sigReadCallback();
void sigReleaseCallback();

bool checkInstPush();


#endif