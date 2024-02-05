# tinyllm

minimal transformer inference in c. ~400 lines total.

## quick start

```bash
make
./download.sh
./tinyllm models/stories15M.bin "Once upon a time"
```

## usage

```
./tinyllm <model.bin> [prompt] [-t temp] [-p topp] [-n steps] [-s seed] [-z tokenizer]
```

## what it does

runs llama-style models (tinyllamas, llama2.c format). implements:
- rmsnorm
- rotary position embeddings (rope)
- grouped query attention (gqa)
- swiglu ffn
- kv cache
- top-p sampling
- bpe tokenizer

no dependencies except libc and libm.
