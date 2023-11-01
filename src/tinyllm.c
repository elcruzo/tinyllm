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
