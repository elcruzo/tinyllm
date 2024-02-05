#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/tinyllm.h"
#include "../include/tokenizer.h"

int main(int argc, char **argv) {
    char *model = NULL, *tokenizer_path = "tokenizer.bin", *prompt = NULL;
    float temp = 1.0f, topp = 0.9f;
    int steps = 256;
    unsigned long long seed = 0;
    
    if (argc < 2) { printf("usage: %s <model.bin> [prompt] [-t temp] [-p topp] [-n steps]\n", argv[0]); return 1; }
    model = argv[1];
    for (int i = 2; i < argc; i++) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == 't') temp = atof(argv[++i]);
            else if (argv[i][1] == 'p') topp = atof(argv[++i]);
            else if (argv[i][1] == 'n') steps = atoi(argv[++i]);
            else if (argv[i][1] == 's') seed = atoll(argv[++i]);
            else if (argv[i][1] == 'z') tokenizer_path = argv[++i];
        } else if (!prompt) prompt = argv[i];
    }
    if (!seed) seed = time(NULL);
    
    FILE *f = fopen(model, "rb");
    if (!f) { printf("can't open %s\n", model); return 1; }
    
    Config cfg;
    fread(&cfg, sizeof(Config), 1, f);
    int shared = cfg.vocab_size > 0;
    cfg.vocab_size = abs(cfg.vocab_size);
    
    int kv = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
    Weights w;
    w.token_embedding = malloc(cfg.vocab_size * cfg.dim * sizeof(float));
    w.rms_att_weight = malloc(cfg.n_layers * cfg.dim * sizeof(float));
    w.rms_ffn_weight = malloc(cfg.n_layers * cfg.dim * sizeof(float));
    w.wq = malloc(cfg.n_layers * cfg.dim * cfg.dim * sizeof(float));
    w.wk = malloc(cfg.n_layers * cfg.dim * kv * sizeof(float));
    w.wv = malloc(cfg.n_layers * cfg.dim * kv * sizeof(float));
    w.wo = malloc(cfg.n_layers * cfg.dim * cfg.dim * sizeof(float));
    w.w1 = malloc(cfg.n_layers * cfg.hidden_dim * cfg.dim * sizeof(float));
    w.w2 = malloc(cfg.n_layers * cfg.dim * cfg.hidden_dim * sizeof(float));
    w.w3 = malloc(cfg.n_layers * cfg.hidden_dim * cfg.dim * sizeof(float));
    w.rms_final_weight = malloc(cfg.dim * sizeof(float));
    
    fread(w.token_embedding, sizeof(float), cfg.vocab_size * cfg.dim, f);
    fread(w.rms_att_weight, sizeof(float), cfg.n_layers * cfg.dim, f);
    fread(w.wq, sizeof(float), cfg.n_layers * cfg.dim * cfg.dim, f);
    fread(w.wk, sizeof(float), cfg.n_layers * cfg.dim * kv, f);
    fread(w.wv, sizeof(float), cfg.n_layers * cfg.dim * kv, f);
    fread(w.wo, sizeof(float), cfg.n_layers * cfg.dim * cfg.dim, f);
    fread(w.rms_ffn_weight, sizeof(float), cfg.n_layers * cfg.dim, f);
    fread(w.w1, sizeof(float), cfg.n_layers * cfg.hidden_dim * cfg.dim, f);
    fread(w.w2, sizeof(float), cfg.n_layers * cfg.dim * cfg.hidden_dim, f);
    fread(w.w3, sizeof(float), cfg.n_layers * cfg.hidden_dim * cfg.dim, f);
    fread(w.rms_final_weight, sizeof(float), cfg.dim, f);
    w.wcls = shared ? w.token_embedding : malloc(cfg.vocab_size * cfg.dim * sizeof(float));
    if (!shared) fread(w.wcls, sizeof(float), cfg.vocab_size * cfg.dim, f);
    fclose(f);
    
    Tokenizer tok;
    tokenizer_load(&tok, tokenizer_path, cfg.vocab_size);
    
    RunState state;
    malloc_run_state(&state, &cfg);
    
    int *toks = NULL, ntoks = 0;
    if (prompt) {
        toks = malloc((strlen(prompt) + 3) * sizeof(int));
        tokenizer_encode(&tok, prompt, toks, &ntoks);
    }
    
    int token = 1, prev = 0;
    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(&cfg, &w, &state, token, pos);
        int next;
        if (pos < ntoks - 1) next = toks[pos + 1];
        else {
            if (temp == 0) next = argmax(logits, cfg.vocab_size);
            else {
                for (int i = 0; i < cfg.vocab_size; i++) logits[i] /= temp;
                softmax(logits, cfg.vocab_size);
                next = sample_topp(logits, cfg.vocab_size, topp, &seed);
            }
        }
        if (pos >= ntoks - 1) { printf("%s", tokenizer_decode(&tok, prev, next)); fflush(stdout); }
        prev = token; token = next;
        if (token == 2) break;
    }
    printf("\n");
    
    free_run_state(&state);
    tokenizer_free(&tok);
    if (toks) free(toks);
    free(w.token_embedding); free(w.rms_att_weight); free(w.rms_ffn_weight);
    free(w.wq); free(w.wk); free(w.wv); free(w.wo);
    free(w.w1); free(w.w2); free(w.w3); free(w.rms_final_weight);
    if (!shared) free(w.wcls);
    return 0;
}
