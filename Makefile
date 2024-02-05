CC = gcc
CFLAGS = -Wall -O3 -march=native -ffast-math
LDFLAGS = -lm

SRC = src/main.c src/tinyllm.c src/tokenizer.c
TARGET = tinyllm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

debug: CFLAGS = -Wall -g -O0
debug: $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean debug
