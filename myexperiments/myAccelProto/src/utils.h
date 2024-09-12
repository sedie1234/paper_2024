#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include "controller.h"


int16_t stringToMD5(char* string);
int seekBank(int size);
int allocatedBankNum(int size);
int getOutputBankFromInstID(MD5_t ID); // XXXXYYYY -> XXXX : startbank, YYYY : endbank
int getBank(MD5_t ID, int arg);
bool MD5Compare(MD5_t ID1, MD5_t ID2); //if same, true. if not, false.
int64_t getsize(int16_t* shape, int n);
void bankLock(int startbank, int banknum);
void bankUnLock(int startbank, int banknum);
void bankRelease(MD5_t ID, int arg);

#endif