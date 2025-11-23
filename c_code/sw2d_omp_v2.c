// sw2d_omp_v2.c — Optimized OpenMP Shallow Water Simulation
// Optimizations: Ghost Cells (Padding), Aligned Memory, Vectorization-friendly loops, First-touch Init.

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <omp.h>

// Alignment for AVX-512 (64 bytes)
#define ALIGNMENT 64

typedef struct {
    int rows, cols, steps;
    double dx, dy, dt;
    double g, H0, cfl;
    double init_height;
    int threads;
    const char *out_file;
    int save_interval;
    int progress_bar;
} Params;

static double now_sec(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

// Helper to allocate aligned memory
static double* alloc_aligned(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGNMENT, size * sizeof(double)) != 0) return NULL;
    return (double*)ptr;
}

int main(int argc, char** argv) {
    Params P = {
        .rows=200, .cols=200, .steps=2000,
        .dx=1.0, .dy=1.0, .dt=0.0,
        .g=9.81, .H0=1.0, .cfl=0.4,
        .init_height=0.5, .threads=0,
        .out_file=NULL, .save_interval=0, .progress_bar=0
    };

    // Quick arg parsing (kept minimal for brevity, matches benchmark needs)
    static struct option long_opts[] = {
        {"rows", required_argument, 0, 0}, {"cols", required_argument, 0, 0},
        {"steps", required_argument, 0, 0}, {"threads", required_argument, 0, 0},
        {"out", required_argument, 0, 0}, {"save-interval", required_argument, 0, 0},
        {"no-progress", no_argument, 0, 0}, {0,0,0,0}
    };
    int optidx;
    while(1) {
        int c = getopt_long(argc, argv, "", long_opts, &optidx);
        if (c==-1) break;
        if (c==0) {
             const char* on = long_opts[optidx].name;
             if (!strcmp(on,"rows")) P.rows = atoi(optarg);
             else if (!strcmp(on,"cols")) P.cols = atoi(optarg);
             else if (!strcmp(on,"steps")) P.steps = atoi(optarg);
             else if (!strcmp(on,"threads")) P.threads = atoi(optarg);
             else if (!strcmp(on,"out")) P.out_file = optarg;
             else if (!strcmp(on,"save-interval")) P.save_interval = atoi(optarg);
        }
    }

    if (P.threads > 0) omp_set_num_threads(P.threads);

    int R = P.rows;
    int C = P.cols;
    
    // Padding: Add 1 ghost cell on each side (top, bottom, left, right)
    // Physical domain: [1..R][1..C]
    // Allocated size: (R+2) * (C+2)
    int pad_R = R + 2;
    int pad_C = C + 2;
    size_t N = (size_t)pad_R * pad_C;

    double *h = alloc_aligned(N);
    double *u = alloc_aligned(N);
    double *v = alloc_aligned(N);
    double *h_new = alloc_aligned(N);
    double *u_new = alloc_aligned(N);
    double *v_new = alloc_aligned(N);

    // Time step
    double wavespeed = sqrt(P.g * P.H0);
    if (P.dt <= 0.0) P.dt = P.cfl * ((P.dx < P.dy) ? P.dx : P.dy) / wavespeed;

    double inv2dx = 1.0 / (2.0 * P.dx);
    double inv2dy = 1.0 / (2.0 * P.dy);
    double dt = P.dt;
    double g = P.g;
    double H0 = P.H0;

    // --- Initialization (First Touch) ---
    // Initialize computation area [1..R][1..C]
    // Also init ghost cells to 0.0
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < pad_R; r++) {
        for (int c = 0; c < pad_C; c++) {
            int idx = r * pad_C + c;
            h[idx] = 0.0; u[idx] = 0.0; v[idx] = 0.0;
            h_new[idx] = 0.0; u_new[idx] = 0.0; v_new[idx] = 0.0;
        }
    }

    // Apply IC (Disk) to physical domain
    double cx = 0.5 * (C - 1) * P.dx;
    double cy = 0.5 * (R - 1) * P.dy;
    double r2_max = ((C * P.dx) / 8.0) * ((C * P.dx) / 8.0);

    #pragma omp parallel for schedule(static)
    for (int r = 1; r <= R; r++) {
        double y = (r - 1) * P.dy;
        double dy = y - cy;
        for (int c = 1; c <= C; c++) {
            double x = (c - 1) * P.dx;
            double dx = x - cx;
            if (dx*dx + dy*dy <= r2_max) {
                h[r * pad_C + c] = P.init_height;
            }
        }
    }

    // Apply Boundary Conditions (Clamping / Zero Gradient)
    // For zero gradient, we just copy the edge values to ghost cells.
    // Top/Bottom
    #pragma omp parallel for
    for (int c = 1; c <= C; c++) {
        h[0 * pad_C + c] = h[1 * pad_C + c];           // Top Halo
        h[(pad_R-1) * pad_C + c] = h[R * pad_C + c];   // Bottom Halo
        // Same for u, v if needed, but u,v start at 0
    }
    // Left/Right
    #pragma omp parallel for
    for (int r = 1; r <= R; r++) {
        h[r * pad_C + 0] = h[r * pad_C + 1];           // Left Halo
        h[r * pad_C + (pad_C-1)] = h[r * pad_C + C];   // Right Halo
    }

    double start_t = now_sec();

    // Main Loop
    for (int step = 1; step <= P.steps; ++step) {
        
        // 1. Update U and V
        // No if statements. Access h[r][c-1] etc. directly because of padding.
        #pragma omp parallel for schedule(static)
        for (int r = 1; r <= R; r++) {
            // Restrict pointers help vectorization
            double * __restrict__ p_u = &u[r * pad_C];
            double * __restrict__ p_v = &v[r * pad_C];
            double * __restrict__ p_u_new = &u_new[r * pad_C];
            double * __restrict__ p_v_new = &v_new[r * pad_C];
            const double * __restrict__ p_h = &h[r * pad_C];
            const double * __restrict__ p_h_up = &h[(r+1) * pad_C];
            const double * __restrict__ p_h_dn = &h[(r-1) * pad_C];

            #pragma omp simd
            for (int c = 1; c <= C; c++) {
                double dhdx = (p_h[c+1] - p_h[c-1]) * inv2dx;
                double dhdy = (p_h_up[c] - p_h_dn[c]) * inv2dy;
                
                p_u_new[c] = p_u[c] - g * dhdx * dt;
                p_v_new[c] = p_v[c] - g * dhdy * dt;
            }
        }

        // Boundary Conditions for U, V (Ghost Cells)
        // Simplified: Set ghosts to 0 (Reflective/Wall) or Copy (Neumann)
        // Here strictly sticking to Neumann (copy) to match serial logic
        // NOTE: In a real high-perf kernel, we might bake this into the loop or skip it if boundaries are far.
        // For correctness with serial, we re-apply BCs.
        #pragma omp parallel
        {
            // Update Ghost Cells U/V
            #pragma omp for nowait
            for (int c = 1; c <= C; c++) {
                u_new[0*pad_C + c] = u_new[1*pad_C + c];
                u_new[(pad_R-1)*pad_C + c] = u_new[R*pad_C + c];
                v_new[0*pad_C + c] = v_new[1*pad_C + c];
                v_new[(pad_R-1)*pad_C + c] = v_new[R*pad_C + c];
            }
            #pragma omp for
            for (int r = 1; r <= R; r++) {
                u_new[r*pad_C + 0] = u_new[r*pad_C + 1];
                u_new[r*pad_C + (pad_C-1)] = u_new[r*pad_C + C];
                v_new[r*pad_C + 0] = v_new[r*pad_C + 1];
                v_new[r*pad_C + (pad_C-1)] = v_new[r*pad_C + C];
            }
        }

        // 2. Update H
        #pragma omp parallel for schedule(static)
        for (int r = 1; r <= R; r++) {
            double * __restrict__ p_h = &h[r * pad_C];
            double * __restrict__ p_h_new = &h_new[r * pad_C];
            const double * __restrict__ p_u = &u_new[r * pad_C];
            const double * __restrict__ p_v = &v_new[r * pad_C];
            const double * __restrict__ p_v_up = &v_new[(r+1) * pad_C];
            const double * __restrict__ p_v_dn = &v_new[(r-1) * pad_C];

            #pragma omp simd
            for (int c = 1; c <= C; c++) {
                double dudx = (p_u[c+1] - p_u[c-1]) * inv2dx;
                double dvdy = (p_v_up[c] - p_v_dn[c]) * inv2dy;
                
                p_h_new[c] = p_h[c] - H0 * (dudx + dvdy) * dt;
            }
        }

        // Boundary Conditions for H
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int c = 1; c <= C; c++) {
                h_new[0*pad_C + c] = h_new[1*pad_C + c];
                h_new[(pad_R-1)*pad_C + c] = h_new[R*pad_C + c];
            }
            #pragma omp for
            for (int r = 1; r <= R; r++) {
                h_new[r*pad_C + 0] = h_new[r*pad_C + 1];
                h_new[r*pad_C + (pad_C-1)] = h_new[r*pad_C + C];
            }
        }

        // 3. Pointer Swap
        double *tmp;
        tmp = h; h = h_new; h_new = tmp;
        tmp = u; u = u_new; u_new = tmp;
        tmp = v; v = v_new; v_new = tmp;
    }

    double end_t = now_sec();
    fprintf(stderr, "Done V2. Wall=%.3fs\n", end_t - start_t);

    free(h); free(u); free(v); free(h_new); free(u_new); free(v_new);
    return 0;
}