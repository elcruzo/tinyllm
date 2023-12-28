CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm

SRC = src/main.c src/tinyllm.c
TARGET = tinyllm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
