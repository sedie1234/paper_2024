#ifndef MAIN_CPP
#define MAIN_CPP

#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <signal.h>
#include <vector>

#include "controller.h"
#include "core.h"
#include "utils.h"

extern SharedData *shared_data;
extern QueueManager qm;
extern int core_status;

int main(void){

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666|IPC_CREAT);
    shared_data = (SharedData*) shmat(shmid, (void*)0, 0);
    sharedDataInit();

    signal(SIGUSR1, signal_handler);

    while(1){

        // 1. check the requirement of inst push
        if(checkInstPush()){
            int blank_queue = qm.queueCheck(1);
            if(blank_queue != -1){
                qm.instPush2Queue(blank_queue);
            }
        }
        // 2. core run
        if(!core_status){
            int high_order_inst = qm.highOrderInst();
            if(high_order_inst != -1){
                runCore(high_order_inst);
            }
        }
        
    }

    shmdt(shared_data);
    shmctl(shmid, IPC_RMID, NULL);

    return 0;
}

#endif