# Makefile for sw2d — 2D shallow-water simulation
# Targets:
#   make / make all    : Build serial, openmp, and pthread versions (v1 and v2)
#   make clean         : Remove executables and objects

CC      := gcc
# Added -mavx2 and -fno-math-errno to encourage vectorization
CFLAGS  := -O3 -march=native -Wall -Wextra -D_POSIX_C_SOURCE=200809L -mavx2 -fno-math-errno
LDLIBS  := -lm -lpthread -fopenmp

# Target Executable Names
TARGETS := sw2d_serial sw2d_omp sw2d_pthread sw2d_omp_v2 sw2d_pthread_v2

.PHONY: all clean help

all: $(TARGETS)

# Serial Version
sw2d_serial: sw2d_serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# OpenMP Versions
sw2d_omp: sw2d_omp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDLIBS)

sw2d_omp_v2: sw2d_omp_v2.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDLIBS)

# Pthreads Versions
sw2d_pthread: sw2d_pthread.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

sw2d_pthread_v2: sw2d_pthread_v2.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGETS) *.o *.bin

help:
	@echo "Available targets:"
	@echo "  make sw2d_serial      : Build serial baseline"
	@echo "  make sw2d_omp         : Build OpenMP v1"
	@echo "  make sw2d_omp_v2      : Build OpenMP v2 (Optimized)"
	@echo "  make sw2d_pthread     : Build Pthreads v1"
	@echo "  make sw2d_pthread_v2  : Build Pthreads v2 (Optimized)"
	@echo "  make all              : Build all five"