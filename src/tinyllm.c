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
void rope(float* q, float* k, int dim, int head_dim, int pos) {
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
    }
}
