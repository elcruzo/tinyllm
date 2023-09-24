#ifndef TINYLLM_H
#define TINYLLM_H

#include <stdint.h>

// model configuration
typedef struct {
    int dim;        // transformer dimension
    int hidden_dim; // ffn hidden dimension
    int n_layers;   // number of transformer layers
    int n_heads;    // number of attention heads
    int n_kv_heads; // number of kv heads (for GQA)
    int vocab_size; // vocabulary size
    int seq_len;    // max sequence length
} Config;

// model weights
typedef struct {
    float* token_embedding; // (vocab_size, dim)
    float* rms_att_weight;  // (n_layers, dim)
    float* rms_ffn_weight;  // (n_layers, dim)
    // attention weights
    float* wq; // (n_layers, dim, n_heads * head_dim)
    float* wk; // (n_layers, dim, n_kv_heads * head_dim)
    float* wv; // (n_layers, dim, n_kv_heads * head_dim)
    float* wo; // (n_layers, n_heads * head_dim, dim)
    // ffn weights
    float* w1; // (n_layers, hidden_dim, dim)
    float* w2; // (n_layers, dim, hidden_dim)
    float* w3; // (n_layers, hidden_dim, dim)
} Weights;

#endif
