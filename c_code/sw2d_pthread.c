// sw2d_pthread.c — 2D shallow-water (linearized) Pthreads Parallel Version
// Features: Persistent threads, custom barrier (macOS compatible), shared memory structs.

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <errno.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Custom Barrier for macOS Compatibility ---
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

// --- Simulation Data Structures ---

typedef struct {
    int rows, cols, steps;
    double dx, dy, dt;
    double g, H0, cfl;
    double init_height;
    int init_col;
    const char *init_file;
    const char *out_file;
    int save_interval;
    int stats_interval;
    int progress_bar;
    int threads;
} Params;

// Shared data among threads
typedef struct {
    int rows, cols;
    double dx, dy, dt, g, H0;
    double *h, *u, *v;
    double *h_new, *u_new, *v_new;
    int steps;
    Barrier *bar;
} SharedContext;

// Thread specific data
typedef struct {
    int id;
    int r_start, r_end; // Row block assigned to this thread
    SharedContext *ctx;
} ThreadArgs;

static double now_sec(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static inline int idx(int r, int c, int cols) { return r*cols + c; }
static inline int clampi(int x, int lo, int hi) {
    if (x < lo) return lo; if (x > hi) return hi; return x;
}
static inline double getH(const double *A, int r, int c, int R, int C) {
    r = clampi(r, 0, R-1); c = clampi(c, 0, C-1);
    return A[idx(r,c,C)];
}

// --- Worker Function ---
void* worker_thread(void *arg) {
    ThreadArgs *ta = (ThreadArgs*)arg;
    SharedContext *ctx = ta->ctx;
    int R = ctx->rows;
    int C = ctx->cols;
    double inv2dx = 1.0/(2.0*ctx->dx);
    double inv2dy = 1.0/(2.0*ctx->dy);

    for (int step = 1; step <= ctx->steps; ++step) {
        
        // Phase 1: Update U, V
        // Read pointers locally inside the loop because they swap every iteration
        double *h = ctx->h;
        double *u = ctx->u;
        double *v = ctx->v;
        double *u_new = ctx->u_new;
        double *v_new = ctx->v_new;
        
        for (int r = ta->r_start; r < ta->r_end; ++r) {
            for (int c = 0; c < C; ++c) {
                double hL = getH(h,r,c-1,R,C), hR = getH(h,r,c+1,R,C);
                double hD = getH(h,r-1,c,R,C), hU = getH(h,r+1,c,R,C);
                double dhdx = (hR - hL) * inv2dx;
                double dhdy = (hU - hD) * inv2dy;
                int id = idx(r,c,C);
                u_new[id] = u[id] + (-ctx->g * dhdx) * ctx->dt;
                v_new[id] = v[id] + (-ctx->g * dhdy) * ctx->dt;
            }
        }
        
        // Wait for all threads to finish U,V update
        barrier_wait(ctx->bar);

        // Phase 2: Update H
        // Note: u_new/v_new are fully populated now
        double *h_new = ctx->h_new;

        for (int r = ta->r_start; r < ta->r_end; ++r) {
            for (int c = 0; c < C; ++c) {
                double uL = getH(u_new,r,c-1,R,C), uR = getH(u_new,r,c+1,R,C);
                double vD = getH(v_new,r-1,c,R,C), vU = getH(v_new,r+1,c,R,C);
                double dudx = (uR - uL) * inv2dx;
                double dvdy = (vU - vD) * inv2dy;
                int id = idx(r,c,C);
                h_new[id] = h[id] + (-ctx->H0 * (dudx + dvdy)) * ctx->dt;
            }
        }

        // Wait for all threads to finish H update
        barrier_wait(ctx->bar);

        // Phase 3: Master Swap
        // Only thread 0 swaps the pointers in the shared context
        if (ta->id == 0) {
            double *tmp;
            tmp = ctx->h; ctx->h = ctx->h_new; ctx->h_new = tmp;
            tmp = ctx->u; ctx->u = ctx->u_new; ctx->u_new = tmp;
            tmp = ctx->v; ctx->v = ctx->v_new; ctx->v_new = tmp;
        }

        // Wait for swap to complete before starting next step
        barrier_wait(ctx->bar);
    }
    return NULL;
}

// --- Helper IO Functions ---

static unsigned char* slurp(const char* path, size_t* out_sz) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f); if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    unsigned char* buf = (unsigned char*)malloc((size_t)n);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (rd != (size_t)n) { free(buf); return NULL; }
    *out_sz = rd; return buf;
}

static int load_init(const char* path, int* rows, int* cols,
                     double** h, double** u, double** v) {
    size_t sz = 0; unsigned char* buf = slurp(path, &sz);
    if (!buf) { fprintf(stderr, "[error] Cannot read init '%s': %s\n", path, strerror(errno)); return -1; }
    if (sz < 8) { fprintf(stderr, "[error] Init too small.\n"); free(buf); return -1; }
    const int32_t* i32 = (const int32_t*)buf;
    int R = i32[0], C = i32[1];
    if (R <= 0 || C <= 0) { fprintf(stderr, "[error] Bad dims in init.\n"); free(buf); return -1; }
    size_t off = 2*sizeof(int32_t);
    size_t need = (size_t)R*(size_t)C*sizeof(double);
    size_t rem = sz - off;

    bool has_all = false;
    if (rem == need) has_all = false;
    else if (rem == 3*need) has_all = true;
    else { fprintf(stderr, "[error] Init payload size mismatch.\n"); free(buf); return -1; }

    double *H = (double*)malloc(need);
    double *U = (double*)calloc((size_t)R*(size_t)C, sizeof(double));
    double *V = (double*)calloc((size_t)R*(size_t)C, sizeof(double));
    if (!H || !U || !V) { fprintf(stderr, "[error] OOM.\n"); free(H); free(U); free(V); free(buf); return -1; }

    memcpy(H, buf + off, need);
    if (has_all) {
        memcpy(U, buf + off + need, need);
        memcpy(V, buf + off + 2*need, need);
    }
    free(buf);
    *rows = R; *cols = C; *h = H; *u = U; *v = V;
    return 0;
}

static void usage(const char* prog) {
    fprintf(stderr,
      "Usage: %s [options]\n"
      "  --rows INT            grid rows (N) [200]\n"
      "  --cols INT            grid cols (M) [200]\n"
      "  --steps INT           number of time steps [2000]\n"
      "  --dx DOUBLE           cell size x [1.0]\n"
      "  --dy DOUBLE           cell size y [1.0]\n"
      "  --dt DOUBLE           time step (<=0 => CFL) [auto]\n"
      "  --g DOUBLE            gravity [9.81]\n"
      "  --H0 DOUBLE           mean depth [1.0]\n"
      "  --cfl DOUBLE          CFL number [0.4]\n"
      "  --height DOUBLE       displaced column height if no init [0.5]\n"
      "  --col INT             (legacy) column index\n"
      "  --init PATH           optional binary init/prior\n"
      "  --out PATH            output movie filename\n"
      "  --save-interval INT   save every k steps (0 disables)\n"
      "  --stats-interval INT  stats update every k steps\n"
      "  --threads INT         Number of threads (default 4)\n"
      "  --no-progress         disable progress bar\n"
      "  --help\n", prog);
}

static void draw_progress(int step, int steps, double ema_upd, double ema_GBps) {
    const int width = 40;
    double frac = steps>0? (double)step/(double)steps : 1.0;
    if (frac < 0) frac = 0; if (frac > 1) frac = 1;
    int filled = (int)llround(frac * width);
    fprintf(stderr, "\r[");
    for (int i=0;i<width;i++) fputc(i<filled?'=':' ', stderr);
    fprintf(stderr, "] %6.2f%%  upd/s~%.2e  BW~%.2f GB/s", 100.0*frac, ema_upd, ema_GBps);
    if (step == steps) fputc('\n', stderr);
    fflush(stderr);
}

// --- Main ---

int main(int argc, char** argv) {
    Params P = {
        .rows=200, .cols=200, .steps=2000,
        .dx=1.0, .dy=1.0, .dt=0.0,
        .g=9.81, .H0=1.0, .cfl=0.4,
        .init_height=0.5, .init_col=-1,
        .init_file=NULL, .out_file=NULL,
        .save_interval=0,
        .stats_interval=100, .progress_bar=1,
        .threads=4 // Default for pthreads
    };

    static struct option long_opts[] = {
        {"rows", required_argument, 0, 0},
        {"cols", required_argument, 0, 0},
        {"steps", required_argument, 0, 0},
        {"dx", required_argument, 0, 0},
        {"dy", required_argument, 0, 0},
        {"dt", required_argument, 0, 0},
        {"g", required_argument, 0, 0},
        {"H0", required_argument, 0, 0},
        {"cfl", required_argument, 0, 0},
        {"height", required_argument, 0, 0},
        {"col", required_argument, 0, 0},
        {"init", required_argument, 0, 0},
        {"out", required_argument, 0, 0},
        {"save-interval", required_argument, 0, 0},
        {"stats-interval", required_argument, 0, 0},
        {"threads", required_argument, 0, 0},
        {"no-progress", no_argument, 0, 0},
        {"help", no_argument, 0, 0},
        {0,0,0,0}
    };
    int optidx;
    while (1) {
        int c = getopt_long(argc, argv, "", long_opts, &optidx);
        if (c == -1) break;
        if (c != 0) continue;
        const char* on = long_opts[optidx].name;
        if      (strcmp(on,"rows")==0) P.rows = atoi(optarg);
        else if (strcmp(on,"cols")==0) P.cols = atoi(optarg);
        else if (strcmp(on,"steps")==0) P.steps = atoi(optarg);
        else if (strcmp(on,"dx")==0) P.dx = atof(optarg);
        else if (strcmp(on,"dy")==0) P.dy = atof(optarg);
        else if (strcmp(on,"dt")==0) P.dt = atof(optarg);
        else if (strcmp(on,"g")==0) P.g = atof(optarg);
        else if (strcmp(on,"H0")==0) P.H0 = atof(optarg);
        else if (strcmp(on,"cfl")==0) P.cfl = atof(optarg);
        else if (strcmp(on,"height")==0) P.init_height = atof(optarg);
        else if (strcmp(on,"col")==0) P.init_col = atoi(optarg);
        else if (strcmp(on,"init")==0) P.init_file = optarg;
        else if (strcmp(on,"out")==0) P.out_file = optarg;
        else if (strcmp(on,"save-interval")==0) P.save_interval = atoi(optarg);
        else if (strcmp(on,"stats-interval")==0) P.stats_interval = atoi(optarg);
        else if (strcmp(on,"threads")==0) P.threads = atoi(optarg);
        else if (strcmp(on,"no-progress")==0) P.progress_bar = 0;
        else if (strcmp(on,"help")==0) { usage(argv[0]); return 0; }
    }

    if (P.threads < 1) P.threads = 1;

    int R=P.rows, C=P.cols;
    size_t N=(size_t)R*(size_t)C;

    // --- Allocation & Init ---
    double *h=NULL, *u=NULL, *v=NULL;
    if (P.init_file) {
        int r2,c2;
        if (load_init(P.init_file, &r2, &c2, &h, &u, &v) != 0) return 1;
        if (r2!=R || c2!=C) {
            fprintf(stderr,"[error] Init dims %dx%d != requested %dx%d\n", r2,c2,R,C);
            free(h); free(u); free(v); return 1;
        }
    } else {
        h = (double*)calloc(N, sizeof(double));
        u = (double*)calloc(N, sizeof(double));
        v = (double*)calloc(N, sizeof(double));
        if (!h||!u||!v) { fprintf(stderr,"[error] OOM\n"); return 1; }
        // Default Circular IC
        const double cx = 0.5 * (C - 1) * P.dx;
        const double cy = 0.5 * (R - 1) * P.dy;
        const double radius = (C * P.dx) / 8.0;
        const double r2_max = radius * radius;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                const double x = c * P.dx;
                const double dx = x - cx;
                const double y = r * P.dy;
                const double dy = y - cy;
                if (dx*dx + dy*dy <= r2_max) h[idx(r,c,C)] = P.init_height;
                else h[idx(r,c,C)] = 0.0;
            }
        }
    }

    double *h_new = (double*)malloc(N*sizeof(double));
    double *u_new = (double*)malloc(N*sizeof(double));
    double *v_new = (double*)malloc(N*sizeof(double));
    if (!h_new||!u_new||!v_new) { fprintf(stderr,"[error] OOM\n"); return 1; }

    const double wavespeed = sqrt(P.g * P.H0);
    if (P.dt <= 0.0) {
        double dmin = (P.dx < P.dy) ? P.dx : P.dy;
        P.dt = P.cfl * dmin / wavespeed;
    }

    // --- Output File Prep (Main Thread Only) ---
    FILE* fout=NULL;
    int32_t nframes = 0;
    if (P.out_file && P.save_interval > 0) {
        fout = fopen(P.out_file, "wb");
        if (!fout) { fprintf(stderr,"[error] open out '%s': %s\n", P.out_file, strerror(errno)); return 1; }
        const char magic[4] = {'S','W','2','D'};
        uint32_t version = 1u;
        uint32_t flags = 0x7u;
        int32_t rows = R, cols = C;
        int32_t save_int = P.save_interval;
        double dxv = P.dx, dyv = P.dy, dtv = P.dt, gv = P.g, H0v = P.H0;
        fwrite(magic, 1, 4, fout);
        fwrite(&version, sizeof(uint32_t), 1, fout);
        fwrite(&flags, sizeof(uint32_t), 1, fout);
        fwrite(&rows, sizeof(int32_t), 1, fout);
        fwrite(&cols, sizeof(int32_t), 1, fout);
        int32_t nframes_placeholder = 0;
        fwrite(&nframes_placeholder, sizeof(int32_t), 1, fout);
        fwrite(&save_int, sizeof(int32_t), 1, fout);
        fwrite(&dxv, sizeof(double), 1, fout);
        fwrite(&dyv, sizeof(double), 1, fout);
        fwrite(&dtv, sizeof(double), 1, fout);
        fwrite(&gv, sizeof(double), 1, fout);
        fwrite(&H0v, sizeof(double), 1, fout);

        size_t count = (size_t)R*(size_t)C;
        fwrite(h, sizeof(double), count, fout);
        fwrite(u, sizeof(double), count, fout);
        fwrite(v, sizeof(double), count, fout);
        nframes = 1;
    }

    // --- Thread Launch ---
    Barrier bar;
    barrier_init(&bar, P.threads);

    SharedContext ctx = {
        .rows = R, .cols = C,
        .dx = P.dx, .dy = P.dy, .dt = P.dt, .g = P.g, .H0 = P.H0,
        .h = h, .u = u, .v = v,
        .h_new = h_new, .u_new = u_new, .v_new = v_new,
        .steps = P.steps,
        .bar = &bar
    };

    pthread_t *tids = malloc(P.threads * sizeof(pthread_t));
    ThreadArgs *targs = malloc(P.threads * sizeof(ThreadArgs));

    int rows_per_thread = R / P.threads;
    int extra = R % P.threads;
    int current_row = 0;

    double t0 = now_sec();

    for (int i=0; i<P.threads; ++i) {
        targs[i].id = i;
        targs[i].ctx = &ctx;
        targs[i].r_start = current_row;
        int r_count = rows_per_thread + (i < extra ? 1 : 0);
        targs[i].r_end = current_row + r_count;
        current_row += r_count;
        pthread_create(&tids[i], NULL, worker_thread, &targs[i]);
    }

    // --- Wait for threads to finish ---
    for (int i=0; i<P.threads; ++i) {
        pthread_join(tids[i], NULL);
    }

    double t1 = now_sec();

    // --- Post-Simulation I/O (FIX for verification) ---
    // After threads join, the simulation state is in ctx.h (due to pointer swaps).
    // We must refresh our local pointers from the context to see the final result.
    h = ctx.h;
    u = ctx.u;
    v = ctx.v;
    h_new = ctx.h_new;
    u_new = ctx.u_new;
    v_new = ctx.v_new;

    // Write the final frame if needed (matches serial behavior for --save-interval=steps)
    // Logic: If save_interval > 0, we typically want at least the start and end states
    // if the interval aligns. The serial code writes at step % interval == 0.
    // Since we just finished step = P.steps, if P.steps % P.save_interval == 0, we write.
    if (fout && P.save_interval > 0 && (P.steps % P.save_interval == 0)) {
        size_t count = (size_t)R*(size_t)C;
        fwrite(h, sizeof(double), count, fout);
        fwrite(u, sizeof(double), count, fout);
        fwrite(v, sizeof(double), count, fout);
        nframes++;
    }

    // Finalize Header
    if (fout) {
        long endpos = ftell(fout);
        fseek(fout, 20L, SEEK_SET);
        fwrite(&nframes, sizeof(int32_t), 1, fout);
        fseek(fout, endpos, SEEK_SET);
        fclose(fout);
    }

    // Stats
    double wall = t1 - t0;
    double total_updates = (double)R*(double)C*(double)P.steps;
    double upd_s = (wall>0)? total_updates/wall : 0.0;
    const double bytes_per_cell_update_est = 3.0 * (5+1) * 8.0;
    double GBps = (wall>0)? (total_updates*bytes_per_cell_update_est)/(wall*1e9) : 0.0;
    
    fprintf(stderr, "\nDone. Wall=%.3fs  Updates=%.3e  Updates/s=%.2e  Apparent BW=%.2f GB/s\n",
            wall, total_updates, upd_s, GBps);

    // Free memory (using the pointers from ctx which are the current valid heads)
    // Note: we must free the original blocks, but since we swapped them around, 
    // we just need to free both sets. Pointer equality check prevents double free if needed,
    // but here we have two distinct sets of arrays always.
    free(h); free(u); free(v);
    free(h_new); free(u_new); free(v_new);
    free(tids); free(targs);
    return 0;
}