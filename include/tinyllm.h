#ifndef TINYLLM_H
#define TINYLLM_H

#include <stdint.h>

// model configuration
typedef struct {
    int dim;        // transformer dimension
    int hidden_dim; // ffn hidden dimension
    int n_layers;   // number of transformer layers
    int n_heads;    // number of attention heads
} Config;

#endif
