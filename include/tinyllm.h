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
    // final rmsnorm and classifier
    float* rms_final_weight; // (dim,)
    float* wcls;             // (vocab_size, dim) - output classifier
} Weights;

// runtime state during inference
typedef struct {
    float* x;      // activation at current position (dim,)
    float* xb;     // same, but inside residual branch (dim,)
    float* xb2;    // additional buffer for ffn residual (dim,)
    float* hb;     // buffer for hidden dim in ffn (hidden_dim,)
    float* hb2;    // buffer for hidden dim in ffn (hidden_dim,)
    float* q;      // query vector (dim,)
    float* k;      // key vector (kv_dim,)
    float* v;      // value vector (kv_dim,)
    float* att;    // attention scores (n_heads, seq_len)
} RunState;

#endif
