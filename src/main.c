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
    
    // parse optional arguments
    for (int i = 2; i < argc; i++) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == 't' && i + 1 < argc) {
                temperature = atof(argv[++i]);
            } else if (argv[i][1] == 'p' && i + 1 < argc) {
                topp = atof(argv[++i]);
            } else if (argv[i][1] == 'n' && i + 1 < argc) {
                steps = atoi(argv[++i]);
            }
        } else if (prompt == NULL) {
            prompt = argv[i];
        }
    }
    
    // load model file
    FILE* file = fopen(model_path, "rb");
    if (!file) {
        printf("error: couldn't open model file %s\n", model_path);
        return 1;
    }
    
    // read config
    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) {
        printf("error: failed to read config\n");
        fclose(file);
        return 1;
    }
    
    printf("model: dim=%d, layers=%d, heads=%d, vocab=%d\n",
           config.dim, config.n_layers, config.n_heads, config.vocab_size);
    
    // allocate weights
    Weights weights;
    int head_dim = config.dim / config.n_heads;
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    
    weights.token_embedding = malloc(config.vocab_size * config.dim * sizeof(float));
    weights.rms_att_weight = malloc(config.n_layers * config.dim * sizeof(float));
    weights.rms_ffn_weight = malloc(config.n_layers * config.dim * sizeof(float));
    weights.wq = malloc(config.n_layers * config.dim * config.dim * sizeof(float));
    weights.wk = malloc(config.n_layers * config.dim * kv_dim * sizeof(float));
    weights.wv = malloc(config.n_layers * config.dim * kv_dim * sizeof(float));
    weights.wo = malloc(config.n_layers * config.dim * config.dim * sizeof(float));
    weights.w1 = malloc(config.n_layers * config.hidden_dim * config.dim * sizeof(float));
    weights.w2 = malloc(config.n_layers * config.dim * config.hidden_dim * sizeof(float));
    weights.w3 = malloc(config.n_layers * config.hidden_dim * config.dim * sizeof(float));
    weights.rms_final_weight = malloc(config.dim * sizeof(float));
    weights.wcls = malloc(config.vocab_size * config.dim * sizeof(float));
    
    // allocate run state
    RunState state;
    malloc_run_state(&state, &config);
    
    // simple tokenization (just use token ids directly for now)
    int* prompt_tokens = NULL;
    int num_prompt_tokens = 0;
    if (prompt != NULL) {
        // for simplicity, assume prompt is space-separated token ids
        // real implementation would use a proper tokenizer
        prompt_tokens = malloc(strlen(prompt) * sizeof(int));
        char* token = strtok(prompt, " ");
        while (token != NULL) {
            prompt_tokens[num_prompt_tokens++] = atoi(token);
            token = strtok(NULL, " ");
        }
    }
    
    // generation loop
    unsigned long long rng_state = 12345;
    int token = 1; // start with BOS token
    int pos = 0;
    
    while (pos < steps) {
        // forward pass
        float* logits = forward(&config, &weights, &state, token, pos);
        
        // determine next token
        int next;
        if (pos < num_prompt_tokens) {
            // still in prompt, use prompt token
            next = prompt_tokens[pos];
        } else {
            // sample from logits
            if (temperature == 0.0f) {
                next = argmax(logits, config.vocab_size);
            } else {
                // apply temperature
                for (int i = 0; i < config.vocab_size; i++) {
                    logits[i] /= temperature;
                }
                softmax(logits, config.vocab_size);
                next = sample_topp(logits, config.vocab_size, topp, &rng_state);
            }
        }
        
        pos++;
        token = next;
        
        // output token (just print id for now)
        printf("%d ", token);
        fflush(stdout);
    }
    printf("\n");
    
    fclose(file);
    
    return 0;
}
