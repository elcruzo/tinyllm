#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tinyllm.h"

int main(int argc, char** argv) {
    // default parameters
    char* model_path = NULL;
    char* prompt = NULL;
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    
    // parse arguments
    if (argc < 2) {
        printf("usage: %s <model> [prompt] [-t temp] [-p topp] [-n steps]\n", argv[0]);
        return 1;
    }
    
    model_path = argv[1];
    
    return 0;
}
