CC = gcc
CFLAGS = -Wall -O3 -march=native
LDFLAGS = -lm

SRC = src/main.c src/tinyllm.c
TARGET = tinyllm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
