#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
    char **vocab;
    float *scores;
    int size, max_len;
    char byte_pieces[512];
} Tokenizer;

void tokenizer_load(Tokenizer *t, char *path, int vocab_size);
void tokenizer_free(Tokenizer *t);
char *tokenizer_decode(Tokenizer *t, int prev, int token);
void tokenizer_encode(Tokenizer *t, char *text, int *tokens, int *n);

#endif
