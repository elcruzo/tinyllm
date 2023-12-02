#include <stdio.h>
#include <stdlib.h>
#include "../include/tinyllm.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model>\n", argv[0]);
        return 1;
    }
    return 0;
}
