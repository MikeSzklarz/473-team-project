// sw2d.c — 2D shallow-water (linearized) with h,u,v output per frame and rich header.
// Discretization: forward Euler in time, centered differences in space.
// Boundaries: zero-gradient (Neumann) via clamped neighbor access.
//
// Default IC (when no --init): circular displaced water column (disk) centered on the grid,
// with diameter = 1/4 of the physical domain width (cols * dx). Height inside the disk = --height.
//
// Binary INIT/PRIOR file (little-endian):
//   int32 rows, int32 cols,
//   then either rows*cols doubles (h) or 3*rows*cols doubles (h,u,v)
//
// Binary MOVIE/OUTPUT file (little-endian):
//   Header layout (fixed, little-endian)
//     char   magic[4] = "SW2D"
//     uint32 version  = 1
//     uint32 flags    = 0x7  // bit0: has_h, bit1: has_u, bit2: has_v
//     int32  rows
//     int32  cols
//     int32  nframes  (filled-in at end; includes initial frame if saved)
//     int32  save_interval
//     double dx, dy, dt, g, H0
//   Frames (nframes of):
//     double h[rows*cols], double u[rows*cols], double v[rows*cols]
//
// Compile: cc -O3 -march=native -Wall -Wextra -o sw2d sw2d.c -lm
//
// Example: ./sw2d --rows 200 --cols 200 --steps 5000 --height 0.5 --save-interval 50 --out movie.bin

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int rows, cols, steps;
    double dx, dy, dt;
    double g, H0, cfl;
    // init condition if no init file
    double init_height;
    int init_col;  // kept for backward-compat; unused by circular IC
    // I/O
    const char *init_file; // optional prior state (h or h,u,v)
    const char *out_file;  // movie output (h,u,v per frame)
    int save_interval;     // save every k steps (0 => disabled)
    // stats
    int stats_interval;    // print/refresh every k steps
    int progress_bar;      // 1 show progress bar
} Params;

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

// Slurp file
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

// Load init/prior state (allocates h,u,v). Returns 0 on success.
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
      "  --col INT             (legacy) column index (unused with circular IC)\n"
      "  --init PATH           optional binary init/prior (h or h,u,v)\n"
      "  --out PATH            output movie filename (h,u,v per frame)\n"
      "  --save-interval INT   save every k steps (0 disables) [0]\n"
      "  --stats-interval INT  stats update every k steps [100]\n"
      "  --no-progress         disable progress bar\n"
      "  --help\n", prog);
}

// Progress bar
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

int main(int argc, char** argv) {
    Params P = {
        .rows=200, .cols=200, .steps=2000,
        .dx=1.0, .dy=1.0, .dt=0.0,
        .g=9.81, .H0=1.0, .cfl=0.4,
        .init_height=0.5, .init_col=-1,
        .init_file=NULL, .out_file=NULL,
        .save_interval=0,
        .stats_interval=100, .progress_bar=1
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
        {"col", required_argument, 0, 0},       // kept for backward compat
        {"init", required_argument, 0, 0},
        {"out", required_argument, 0, 0},
        {"save-interval", required_argument, 0, 0},
        {"stats-interval", required_argument, 0, 0},
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
        else if (strcmp(on,"col")==0) P.init_col = atoi(optarg); // legacy
        else if (strcmp(on,"init")==0) P.init_file = optarg;
        else if (strcmp(on,"out")==0) P.out_file = optarg;
        else if (strcmp(on,"save-interval")==0) P.save_interval = atoi(optarg);
        else if (strcmp(on,"stats-interval")==0) P.stats_interval = atoi(optarg);
        else if (strcmp(on,"no-progress")==0) P.progress_bar = 0;
        else if (strcmp(on,"help")==0) { usage(argv[0]); return 0; }
    }

    if (P.rows <= 0 || P.cols <= 0 || P.steps < 0) { usage(argv[0]); return 1; }
    if (P.save_interval < 0) P.save_interval = 0;
    if (P.stats_interval <= 0) P.stats_interval = 100;

    int R=P.rows, C=P.cols;
    size_t N=(size_t)R*(size_t)C;

    // State arrays
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

        // --- New default IC: circular displaced column (disk) ---
        // Physical center:
        const double cx = 0.5 * (C - 1) * P.dx;
        const double cy = 0.5 * (R - 1) * P.dy;
        // Requested: diameter = 1/4 of physical width => radius = width/8
        const double radius = (C * P.dx) / 8.0;
        const double r2_max = radius * radius;

        for (int r = 0; r < R; ++r) {
            const double y = r * P.dy;
            for (int c = 0; c < C; ++c) {
                const double x = c * P.dx;
                const double dx = x - cx;
                const double dy = y - cy;
                const double d2 = dx*dx + dy*dy;
                if (d2 <= r2_max) {
                    h[idx(r,c,C)] = P.init_height; // flat top inside the disk
                } else {
                    h[idx(r,c,C)] = 0.0;          // flat (zero perturbation) elsewhere
                }
            }
        }
        // u,v start at 0
    }

    double *h_new = (double*)malloc(N*sizeof(double));
    double *u_new = (double*)malloc(N*sizeof(double));
    double *v_new = (double*)malloc(N*sizeof(double));
    if (!h_new||!u_new||!v_new) { fprintf(stderr,"[error] OOM\n"); return 1; }

    // Time step from CFL if not provided
    const double wavespeed = sqrt(P.g * P.H0);
    if (P.dt <= 0.0) {
        double dmin = (P.dx < P.dy) ? P.dx : P.dy;
        P.dt = P.cfl * dmin / wavespeed;
    }

    // Output
    FILE* fout=NULL;
    int32_t nframes = 0;
    if (P.out_file && P.save_interval > 0) {
        fout = fopen(P.out_file, "wb");
        if (!fout) { fprintf(stderr,"[error] open out '%s': %s\n", P.out_file, strerror(errno)); return 1; }

        // Header
        const char magic[4] = {'S','W','2','D'};
        uint32_t version = 1u;
        uint32_t flags = 0x7u; // has h,u,v
        int32_t rows = R, cols = C;
        int32_t save_int = P.save_interval;
        double dxv = P.dx, dyv = P.dy, dtv = P.dt, gv = P.g, H0v = P.H0;

        fwrite(magic, 1, 4, fout);
        fwrite(&version, sizeof(uint32_t), 1, fout);
        fwrite(&flags, sizeof(uint32_t), 1, fout);
        fwrite(&rows, sizeof(int32_t), 1, fout);
        fwrite(&cols, sizeof(int32_t), 1, fout);
        int32_t nframes_placeholder = 0; // fill at end
        long nframes_pos = ftell(fout);
        fwrite(&nframes_placeholder, sizeof(int32_t), 1, fout);
        fwrite(&save_int, sizeof(int32_t), 1, fout);
        fwrite(&dxv, sizeof(double), 1, fout);
        fwrite(&dyv, sizeof(double), 1, fout);
        fwrite(&dtv, sizeof(double), 1, fout);
        fwrite(&gv, sizeof(double), 1, fout);
        fwrite(&H0v, sizeof(double), 1, fout);

        // Save initial frame (step 0)
        size_t count = (size_t)R*(size_t)C;
        if (fwrite(h, sizeof(double), count, fout) != count) { fprintf(stderr,"[error] write frame h\n"); return 1; }
        if (fwrite(u, sizeof(double), count, fout) != count) { fprintf(stderr,"[error] write frame u\n"); return 1; }
        if (fwrite(v, sizeof(double), count, fout) != count) { fprintf(stderr,"[error] write frame v\n"); return 1; }
        nframes = 1;

        (void)nframes_pos; // kept for clarity; we use fixed offset below
    }

    // Perf model
    const double bytes_per_cell_update_est = 3.0 /*h,u,v*/ * (5+1) /*~5 reads + 1 write*/ * 8.0; // rough
    double t0 = now_sec(), last_t = t0;
    double ema_updates = 0.0, ema_GBps = 0.0, ema_alpha = 0.2;

    const double inv2dx = 1.0/(2.0*P.dx), inv2dy = 1.0/(2.0*P.dy);
    const double H0 = P.H0, g = P.g, dt = P.dt;

    for (int step=1; step<=P.steps; ++step) {
        // u,v from h gradients
        for (int r=0;r<R;++r) {
            for (int c=0;c<C;++c) {
                double hL = getH(h,r,c-1,R,C), hR = getH(h,r,c+1,R,C);
                double hD = getH(h,r-1,c,R,C), hU = getH(h,r+1,c,R,C);
                double dhdx = (hR - hL) * inv2dx;
                double dhdy = (hU - hD) * inv2dy;
                int id = idx(r,c,C);
                u_new[id] = u[id] + (-g * dhdx)*dt;
                v_new[id] = v[id] + (-g * dhdy)*dt;
            }
        }
        // h from divergence of u,v
        for (int r=0;r<R;++r) {
            for (int c=0;c<C;++c) {
                double uL = getH(u_new,r,c-1,R,C), uR = getH(u_new,r,c+1,R,C);
                double vD = getH(v_new,r-1,c,R,C), vU = getH(v_new,r+1,c,R,C);
                double dudx = (uR - uL) * inv2dx;
                double dvdy = (vU - vD) * inv2dy;
                int id = idx(r,c,C);
                h_new[id] = h[id] + (-H0 * (dudx + dvdy))*dt;
            }
        }

        // swap
        double *tmp; tmp=h; h=h_new; h_new=tmp;
        tmp=u; u=u_new; u_new=tmp;
        tmp=v; v=v_new; v_new=tmp;

        // Save frame?
        if (fout && (step % P.save_interval == 0)) {
            size_t count = (size_t)R*(size_t)C;
            if (fwrite(h, sizeof(double), count, fout) != count) { fprintf(stderr,"[error] write frame h\n"); return 1; }
            if (fwrite(u, sizeof(double), count, fout) != count) { fprintf(stderr,"[error] write frame u\n"); return 1; }
            if (fwrite(v, sizeof(double), count, fout) != count) { fprintf(stderr,"[error] write frame v\n"); return 1; }
            nframes++;
        }

        // Stats/progress
        if (P.progress_bar || (P.stats_interval>0 && (step % P.stats_interval)==0)) {
            double t1 = now_sec(); double dt_wall = t1 - last_t; last_t = t1;
            if (dt_wall > 0) {
                double upd = (double)R*(double)C*(double)((step<P.stats_interval)? step : P.stats_interval);
                double upd_s = upd / dt_wall;
                double GBps = (upd * bytes_per_cell_update_est) / (dt_wall*1e9);
                if (ema_updates<=0.0) { ema_updates=upd_s; ema_GBps=GBps; }
                else {
                    ema_updates = ema_alpha*upd_s + (1.0-ema_alpha)*ema_updates;
                    ema_GBps   = ema_alpha*GBps + (1.0-ema_alpha)*ema_GBps;
                }
                if (P.progress_bar) draw_progress(step, P.steps, ema_updates, ema_GBps);
                else fprintf(stderr,"[%d/%d] upd/s~%.2e  BW~%.2f GB/s\n", step,P.steps,ema_updates,ema_GBps);
            }
        }
    }

    if (fout) {
        // Patch nframes in header at fixed offset:
        // magic(4) + version(4) + flags(4) + rows(4) + cols(4) = 20, next 4 bytes are nframes
        long endpos = ftell(fout);
        fseek(fout, 20L, SEEK_SET);
        fwrite(&nframes, sizeof(int32_t), 1, fout);
        fseek(fout, endpos, SEEK_SET);
        fclose(fout);
    }

    double t1 = now_sec();
    double wall = t1 - t0;
    double total_updates = (double)R*(double)C*(double)P.steps;
    double upd_s = (wall>0)? total_updates/wall : 0.0;
    double GBps = (wall>0)? (total_updates*bytes_per_cell_update_est)/(wall*1e9) : 0.0;
    fprintf(stderr, "\nDone. Wall=%.3fs  Updates=%.3e  Updates/s=%.2e  Apparent BW=%.2f GB/s\n",
            wall, total_updates, upd_s, GBps);

    free(h); free(u); free(v);
    free(h_new); free(u_new); free(v_new);
    return 0;
}

