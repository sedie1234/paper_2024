#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // 바이너리 파일 열기
	char* filename = argv[1];

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Failed to open file for reading.\n");
        return 1;
    }

    // 64바이트 읽을 버퍼 준비
//    unsigned char buffer[1024];
	unsigned char *buffer = (unsigned char*)malloc(0x10000);
    size_t bytesRead = fread(buffer, 1, 0x10000, file);
    
    // 파일에서 읽은 바이트 수가 64보다 작을 경우 확인
    if (bytesRead == 0) {
        printf("Failed to read from file or file is empty.\n");
        fclose(file);
        return 1;
    }

    // 16진수로 출력
    printf("First 64 bytes of the file:\n");
    for (size_t i = 0; i < 16; i++) {
		printf("%f ", *(float *)(buffer + i*4));
//        printf("%02X ", buffer[i]);

        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
	
	printf("...\n");

    for (size_t i = 16; i > 0; i--) {
		printf("%f ", *(float *)(buffer + 0x10000 - i*4));
//        printf("%02X ", buffer[i]);

        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }


    printf("\n");

    // 파일 닫기
    fclose(file);
	free(buffer);
    return 0;
}

