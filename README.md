# tinyllm

minimal transformer inference in c. no frameworks.

runs small language models on cpu. ~500 lines.

## build

```
make
```

## usage

```
./tinyllm model.bin [prompt] [-t temperature] [-p topp] [-n steps]
```

## options

- `-t`: temperature for sampling (default: 1.0)
- `-p`: top-p nucleus sampling threshold (default: 0.9)
- `-n`: number of tokens to generate (default: 256)

## architecture

implements a decoder-only transformer with:
- RMSNorm (instead of LayerNorm)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation in FFN
- Grouped Query Attention (GQA) support
- KV cache for efficient autoregressive generation
