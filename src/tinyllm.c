#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tinyllm.h"

// RMS normalization (used instead of LayerNorm in LLaMA)
void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// softmax function for attention scores
void softmax(float* x, int size) {
    // find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// apply rotary position embedding to q and k vectors
void rope(float* q, float* k, int dim, int kv_dim, int head_dim, int pos) {
    for (int i = 0; i < dim; i += 2) {
        int head_i = i % head_dim;
        float freq = 1.0f / powf(10000.0f, head_i / (float)head_dim);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        // rotate q
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * fcr - q1 * fci;
        q[i + 1] = q0 * fci + q1 * fcr;
        // rotate k (only up to kv_dim)
        if (i < kv_dim) {
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i]     = k0 * fcr - k1 * fci;
            k[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

// argmax sampling - returns index of max value
int argmax(float* v, int n) {
    int max_i = 0;
    float max_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_val) {
            max_val = v[i];
            max_i = i;
        }
    }
    return max_i;
}

// random number generator (xorshift)
unsigned long long rng_seed(unsigned long long seed) {
    return seed ^ (seed >> 33);
}

float random_f32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) / (float)0xFFFFFFFFFFFFFFFFull;
}

// top-p (nucleus) sampling
int sample_topp(float* probs, int n, float topp, unsigned long long* rng_state) {
    // sort probabilities in descending order (simple bubble sort for small n)
    int* indices = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;
    
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (probs[indices[j]] < probs[indices[j + 1]]) {
                int tmp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tmp;
            }
        }
    }
    
    // find cutoff index where cumulative prob exceeds topp
    float cumulative = 0.0f;
    int cutoff = n - 1;
    for (int i = 0; i < n; i++) {
        cumulative += probs[indices[i]];
        if (cumulative > topp) {
            cutoff = i;
            break;
        }
    }
    
    // sample from truncated distribution
    float r = random_f32(rng_state) * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= cutoff; i++) {
        cdf += probs[indices[i]];
        if (r < cdf) {
            int result = indices[i];
            free(indices);
            return result;
        }
    }
    
    int result = indices[cutoff];
    free(indices);
    return result;
}

// allocate memory for RunState buffers
void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
}

// free RunState memory
void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// forward pass for one token
float* forward(Config* p, Weights* w, RunState* s, int token, int pos) {
    // convenience variables
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = dim / p->n_heads;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    
    // copy token embedding into x
    float* content_row = w->token_embedding + token * dim;
    for (int i = 0; i < dim; i++) {
        s->x[i] = content_row[i];
    }
    
    // iterate over all transformer layers
    for (int l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        
        // compute q, k, v projections
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
        
        // apply rotary position embedding
        rope(s->q, s->k, dim, kv_dim, head_dim, pos);
        
        // cache k and v for this position
        int loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        for (int i = 0; i < kv_dim; i++) {
            key_cache_row[i] = s->k[i];
            value_cache_row[i] = s->v[i];
        }
        
        // multihead attention
        for (int h = 0; h < p->n_heads; h++) {
            // get query vector for this head
            float* q = s->q + h * head_dim;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps including current
            for (int t = 0; t <= pos; t++) {
                // get key vector for this head at timestep t
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // dot product q . k
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_dim);
                att[t] = score;
            }
            
            // softmax attention scores
            softmax(att, pos + 1);
            
            // weighted sum of values
            float* xb = s->xb + h * head_dim;
            for (int i = 0; i < head_dim; i++) xb[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float a = att[t];
                for (int i = 0; i < head_dim; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        
        // output projection
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        
        // residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
        
        // ffn rmsnorm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
        
        // ffn: w1 and w3 projections (SwiGLU)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        
        // SwiGLU activation: silu(w1(x)) * w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x) = x * sigmoid(x)
            val *= (1.0f / (1.0f + expf(-val)));
            s->hb[i] = val * s->hb2[i];
        }
        
        // w2 projection (down projection)
        matmul(s->xb, s->hb, w->w2 + l * hidden_dim * dim, hidden_dim, dim);
        
        // ffn residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }
    
    // final rmsnorm
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    
    // classifier into logits
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    
    return s->logits;
}
