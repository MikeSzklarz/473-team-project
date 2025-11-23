# Makefile for sw2d — 2D shallow-water simulation
# Targets:
#   make / make all    : Build serial, openmp, and pthread versions
#   make clean         : Remove executables and objects

CC      := gcc
CFLAGS  := -O3 -march=native -Wall -Wextra -D_POSIX_C_SOURCE=200809L
LDLIBS  := -lm -lpthread -fopenmp

# Target Executable Names
TARGETS := sw2d_serial sw2d_omp sw2d_pthread

.PHONY: all clean help

all: $(TARGETS)

# Serial Version
sw2d_serial: sw2d_serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# OpenMP Version
sw2d_omp: sw2d_omp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDLIBS)

# Pthreads Version
sw2d_pthread: sw2d_pthread.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGETS) *.o *.bin

help:
	@echo "Available targets:"
	@echo "  make sw2d_serial   : Build the serial reference implementation"
	@echo "  make sw2d_omp      : Build the OpenMP parallel version"
	@echo "  make sw2d_pthread  : Build the Pthreads parallel version"
	@echo "  make all           : Build all three"
	@echo "  make clean         : Remove all executables"