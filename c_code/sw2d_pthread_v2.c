// sw2d_pthread_v2.c — Optimized Pthreads Shallow Water Simulation
// Optimizations: Ghost Cells, Persistent Threads, Local Barriers, Aligned Memory

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>

#define ALIGNMENT 64

// --- Barrier ---
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int crossing;
    int limit;
} Barrier;

static void barrier_init(Barrier *b, int n) {
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->limit = n;
    b->count = 0;
    b->crossing = 0;
}

static void barrier_wait(Barrier *b) {
    pthread_mutex_lock(&b->mutex);
    int crossing = b->crossing;
    b->count++;
    if (b->count >= b->limit) {
        b->crossing++;
        b->count = 0;
        pthread_cond_broadcast(&b->cond);
    } else {
        while (crossing == b->crossing) {
            pthread_cond_wait(&b->cond, &b->mutex);
        }
    }
    pthread_mutex_unlock(&b->mutex);
}

// --- Data ---
typedef struct {
    int rows, cols, steps;
    double dx, dy, dt, g, H0;
    int pad_R, pad_C;
    double *h, *u, *v;
    double *h_new, *u_new, *v_new;
    Barrier *bar;
} SharedContext;

typedef struct {
    int id;
    int r_start, r_end; // 1-based row indices
    SharedContext *ctx;
} ThreadArgs;

static double* alloc_aligned(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGNMENT, size * sizeof(double)) != 0) return NULL;
    return (double*)ptr;
}

static double now_sec(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

// --- Worker ---
void* worker(void *arg) {
    ThreadArgs *ta = (ThreadArgs*)arg;
    SharedContext *ctx = ta->ctx;
    
    int C = ctx->cols;
    int pad_C = ctx->pad_C;
    double inv2dx = 1.0/(2.0*ctx->dx);
    double inv2dy = 1.0/(2.0*ctx->dy);
    double g = ctx->g;
    double dt = ctx->dt;
    double H0 = ctx->H0;

    for (int step = 1; step <= ctx->steps; ++step) {
        
        // --- Phase 1: U, V ---
        double *h = ctx->h;
        double *u = ctx->u;
        double *v = ctx->v;
        double *u_new = ctx->u_new;
        double *v_new = ctx->v_new;

        for (int r = ta->r_start; r < ta->r_end; ++r) {
            double * __restrict__ p_u = &u[r * pad_C];
            double * __restrict__ p_v = &v[r * pad_C];
            double * __restrict__ p_u_new = &u_new[r * pad_C];
            double * __restrict__ p_v_new = &v_new[r * pad_C];
            const double * __restrict__ p_h = &h[r * pad_C];
            const double * __restrict__ p_h_up = &h[(r+1) * pad_C];
            const double * __restrict__ p_h_dn = &h[(r-1) * pad_C];

            // Auto-vectorizable inner loop
            for (int c = 1; c <= C; ++c) {
                double dhdx = (p_h[c+1] - p_h[c-1]) * inv2dx;
                double dhdy = (p_h_up[c] - p_h_dn[c]) * inv2dy;
                p_u_new[c] = p_u[c] - g * dhdx * dt;
                p_v_new[c] = p_v[c] - g * dhdy * dt;
            }
        }
        
        // Barrier 1: Wait for U,V computation
        barrier_wait(ctx->bar);

        // --- Phase 1.5: U,V Boundaries (Thread 0 does it for simplicity, or parallelize) ---
        // To keep overhead low, let's just have thread 0 do the halos quickly.
        // In a super-optimized version, we'd split this, but halos are small O(N) vs O(N^2).
        if (ta->id == 0) {
            int R = ctx->rows;
            // Top/Bottom
            for (int c = 1; c <= C; c++) {
                u_new[0*pad_C + c] = u_new[1*pad_C + c];
                u_new[(ctx->pad_R-1)*pad_C + c] = u_new[R*pad_C + c];
                v_new[0*pad_C + c] = v_new[1*pad_C + c];
                v_new[(ctx->pad_R-1)*pad_C + c] = v_new[R*pad_C + c];
            }
            // Left/Right
            for (int r = 1; r <= R; r++) {
                u_new[r*pad_C + 0] = u_new[r*pad_C + 1];
                u_new[r*pad_C + (pad_C-1)] = u_new[r*pad_C + C];
                v_new[r*pad_C + 0] = v_new[r*pad_C + 1];
                v_new[r*pad_C + (pad_C-1)] = v_new[r*pad_C + C];
            }
        }
        barrier_wait(ctx->bar);

        // --- Phase 2: H ---
        double *h_new = ctx->h_new;
        
        for (int r = ta->r_start; r < ta->r_end; ++r) {
            double * __restrict__ p_h = &h[r * pad_C];
            double * __restrict__ p_h_new = &h_new[r * pad_C];
            const double * __restrict__ p_u = &u_new[r * pad_C];
            const double * __restrict__ p_v = &v_new[r * pad_C];
            const double * __restrict__ p_v_up = &v_new[(r+1) * pad_C];
            const double * __restrict__ p_v_dn = &v_new[(r-1) * pad_C];

            for (int c = 1; c <= C; ++c) {
                double dudx = (p_u[c+1] - p_u[c-1]) * inv2dx;
                double dvdy = (p_v_up[c] - p_v_dn[c]) * inv2dy;
                p_h_new[c] = p_h[c] - H0 * (dudx + dvdy) * dt;
            }
        }
        
        barrier_wait(ctx->bar);

        // --- Phase 2.5: H Boundaries & Swap ---
        if (ta->id == 0) {
            int R = ctx->rows;
            for (int c = 1; c <= C; c++) {
                h_new[0*pad_C + c] = h_new[1*pad_C + c];
                h_new[(ctx->pad_R-1)*pad_C + c] = h_new[R*pad_C + c];
            }
            for (int r = 1; r <= R; r++) {
                h_new[r*pad_C + 0] = h_new[r*pad_C + 1];
                h_new[r*pad_C + (pad_C-1)] = h_new[r*pad_C + C];
            }

            // Swap
            double *t;
            t = ctx->h; ctx->h = ctx->h_new; ctx->h_new = t;
            t = ctx->u; ctx->u = ctx->u_new; ctx->u_new = t;
            t = ctx->v; ctx->v = ctx->v_new; ctx->v_new = t;
        }
        barrier_wait(ctx->bar);
    }
    return NULL;
}

// --- Main ---
int main(int argc, char** argv) {
    int rows=200, cols=200, steps=2000, threads=4;
    double dx=1.0, dy=1.0, dt=0.0, g=9.81, H0=1.0, cfl=0.4, height=0.5;
    
    // Minimal arg parse
    for(int i=1; i<argc; i++){
        if(!strcmp(argv[i],"--rows")) rows=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cols")) cols=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--steps")) steps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")) threads=atoi(argv[++i]);
    }

    int pad_R = rows + 2;
    int pad_C = cols + 2;
    size_t N = (size_t)pad_R * pad_C;

    double *h = alloc_aligned(N);
    double *u = alloc_aligned(N);
    double *v = alloc_aligned(N);
    double *h_new = alloc_aligned(N);
    double *u_new = alloc_aligned(N);
    double *v_new = alloc_aligned(N);

    // Init (Serial for simplicity in pthread version, normally parallelize for first touch)
    memset(h, 0, N*sizeof(double));
    memset(u, 0, N*sizeof(double));
    memset(v, 0, N*sizeof(double));
    
    double cx = 0.5 * (cols-1)*dx;
    double cy = 0.5 * (rows-1)*dy;
    double r2_max = ((cols*dx)/8.0)*((cols*dx)/8.0);
    
    for(int r=1; r<=rows; r++) {
        for(int c=1; c<=cols; c++) {
            double x = (c-1)*dx - cx;
            double y = (r-1)*dy - cy;
            if (x*x + y*y <= r2_max) h[r*pad_C + c] = height;
        }
    }

    double wavespeed = sqrt(g*H0);
    if(dt <= 0.0) dt = cfl * ((dx<dy)?dx:dy)/wavespeed;

    Barrier bar;
    barrier_init(&bar, threads);

    SharedContext ctx = {
        .rows=rows, .cols=cols, .steps=steps,
        .dx=dx, .dy=dy, .dt=dt, .g=g, .H0=H0,
        .pad_R=pad_R, .pad_C=pad_C,
        .h=h, .u=u, .v=v, .h_new=h_new, .u_new=u_new, .v_new=v_new,
        .bar=&bar
    };

    pthread_t *tids = malloc(threads * sizeof(pthread_t));
    ThreadArgs *args = malloc(threads * sizeof(ThreadArgs));

    int rows_per_thread = rows / threads;
    int extra = rows % threads;
    int current_row = 1;

    double t0 = now_sec();

    for(int i=0; i<threads; i++) {
        args[i].id = i;
        args[i].ctx = &ctx;
        args[i].r_start = current_row;
        int r_count = rows_per_thread + (i < extra ? 1 : 0);
        args[i].r_end = current_row + r_count;
        current_row += r_count;
        pthread_create(&tids[i], NULL, worker, &args[i]);
    }

    for(int i=0; i<threads; i++) pthread_join(tids[i], NULL);

    double t1 = now_sec();
    fprintf(stderr, "Done V2. Wall=%.3fs\n", t1 - t0);

    free(h); free(u); free(v); free(h_new); free(u_new); free(v_new);
    free(tids); free(args);
    return 0;
}