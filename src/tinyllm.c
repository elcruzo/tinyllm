#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tinyllm.h"

void rmsnorm(float *o, float *x, float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = w[i] * ss * x[i];
}

void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void matmul(float *o, float *x, float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float v = 0;
        for (int j = 0; j < n; j++) v += w[i*n + j] * x[j];
        o[i] = v;
    }
}

void rope(float *q, float *k, int dim, int kv_dim, int hd, int pos) {
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (i % hd) / (float)hd);
        float c = cosf(pos * freq), s = sinf(pos * freq);
        float q0 = q[i], q1 = q[i+1];
        q[i] = q0*c - q1*s; q[i+1] = q0*s + q1*c;
        if (i < kv_dim) {
            float k0 = k[i], k1 = k[i+1];
            k[i] = k0*c - k1*s; k[i+1] = k0*s + k1*c;
        }
    }
}

int argmax(float *v, int n) {
    int m = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[m]) m = i;
    return m;
}

float randf(unsigned long long *s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (*s * 0x2545F4914F6CDD1Dull) / (float)0xFFFFFFFFFFFFFFFFull;
}

int sample_topp(float *p, int n, float topp, unsigned long long *rng) {
    int *idx = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-i-1; j++)
            if (p[idx[j]] < p[idx[j+1]]) { int t = idx[j]; idx[j] = idx[j+1]; idx[j+1] = t; }
    float cum = 0; int cut = n-1;
    for (int i = 0; i < n; i++) { cum += p[idx[i]]; if (cum > topp) { cut = i; break; } }
    float r = randf(rng) * cum, cdf = 0;
    for (int i = 0; i <= cut; i++) { cdf += p[idx[i]]; if (r < cdf) { int res = idx[i]; free(idx); return res; } }
    int res = idx[cut]; free(idx); return res;
}

void malloc_run_state(RunState *s, Config *p) {
    int kv = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv, sizeof(float));
    s->v = calloc(kv, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv, sizeof(float));
}

void free_run_state(RunState *s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
}

float *forward(Config *p, Weights *w, RunState *s, int token, int pos) {
    int dim = p->dim, hd = dim / p->n_heads;
    int kv = (dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    
    float *x = w->token_embedding + token * dim;
    for (int i = 0; i < dim; i++) s->x[i] = x[i];
    
    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + l*dim, dim);
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv, dim, kv);
        matmul(s->v, s->xb, w->wv + l*dim*kv, dim, kv);
        rope(s->q, s->k, dim, kv, hd, pos);
        
        int off = l * p->seq_len * kv;
        float *kc = s->key_cache + off + pos*kv, *vc = s->value_cache + off + pos*kv;
        for (int i = 0; i < kv; i++) { kc[i] = s->k[i]; vc[i] = s->v[i]; }
        
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h*hd, *att = s->att + h*p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + off + t*kv + (h/kv_mul)*hd;
                float sc = 0; for (int i = 0; i < hd; i++) sc += q[i] * k[i];
                att[t] = sc / sqrtf(hd);
            }
            softmax(att, pos+1);
            float *xb = s->xb + h*hd;
            for (int i = 0; i < hd; i++) xb[i] = 0;
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + off + t*kv + (h/kv_mul)*hd;
                for (int i = 0; i < hd; i++) xb[i] += att[t] * v[i];
            }
        }
        
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
        
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);
        matmul(s->hb, s->xb, w->w1 + l*dim*p->hidden_dim, dim, p->hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*p->hidden_dim, dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++)
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i]))) * s->hb2[i];
        matmul(s->xb, s->hb, w->w2 + l*p->hidden_dim*dim, p->hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }
    
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
}
