#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char **vocab;
    float *scores;
    int size;
    int max_len;
    char byte_pieces[512];
} Tokenizer;

void tokenizer_load(Tokenizer *t, char *path, int vocab_size) {
    t->size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char*));
    t->scores = malloc(vocab_size * sizeof(float));
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i*2] = (char)i;
        t->byte_pieces[i*2+1] = '\0';
    }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", path); exit(1); }
    fread(&t->max_len, sizeof(int), 1, f);
    for (int i = 0; i < vocab_size; i++) {
        int len;
        fread(&t->scores[i], sizeof(float), 1, f);
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = malloc(len + 1);
        fread(t->vocab[i], 1, len, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

void tokenizer_free(Tokenizer *t) {
    for (int i = 0; i < t->size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
}

char *tokenizer_decode(Tokenizer *t, int prev, int token) {
    char *s = t->vocab[token];
    if (prev == 1 && s[0] == ' ') s++;
    unsigned char b;
    if (sscanf(s, "<0x%02hhX>", &b) == 1) s = t->byte_pieces + b*2;
    return s;
}

int tokenizer_lookup(Tokenizer *t, char *s) {
    for (int i = 0; i < t->size; i++)
        if (!strcmp(s, t->vocab[i])) return i;
    return -1;
}

void tokenizer_encode(Tokenizer *t, char *text, int *tokens, int *n) {
    char *buf = malloc(t->max_len * 2 + 3);
    *n = 0;
    tokens[(*n)++] = 1;
    int sp = tokenizer_lookup(t, " ");
    if (sp != -1 && text[0]) tokens[(*n)++] = sp;
    
    for (char *c = text; *c; c++) {
        buf[0] = *c; buf[1] = '\0';
        int len = 1;
        while ((*(c+1) & 0xC0) == 0x80) { c++; buf[len++] = *c; buf[len] = '\0'; }
        int id = tokenizer_lookup(t, buf);
        if (id != -1) tokens[(*n)++] = id;
        else for (int i = 0; i < len; i++) {
            sprintf(buf, "<0x%02X>", (unsigned char)buf[i]);
            id = tokenizer_lookup(t, buf);
            if (id != -1) tokens[(*n)++] = id;
        }
    }
    
    for (;;) {
        float best = -1e10; int best_id = -1, best_i = -1;
        for (int i = 0; i < *n - 1; i++) {
            sprintf(buf, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = tokenizer_lookup(t, buf);
            if (id != -1 && t->scores[id] > best) { best = t->scores[id]; best_id = id; best_i = i; }
        }
        if (best_i == -1) break;
        tokens[best_i] = best_id;
        for (int i = best_i + 1; i < *n - 1; i++) tokens[i] = tokens[i+1];
        (*n)--;
    }
    free(buf);
}
