import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import logging
import os
import sys
import argparse
import seaborn as sns
import csv
from datetime import datetime

# Configuration
C_CODE_DIR = "../c_code"
RESULTS_ROOT_DIR = "../benchmark_runs"

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def compile_simulation():
    """Compiles the C code using the Makefile in c_code directory."""
    logger.info("Compiling C simulation code...")
    try:
        subprocess.check_call(["make", "-C", C_CODE_DIR], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Compilation successful.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Compilation failed with error code {e.returncode}")
        sys.exit(1)

def run_simulation(exe, rows, threads, steps):
    """Runs the simulation and parses Wall time."""
    # Critical Optimization: Set save-interval to 0 to disable I/O during benchmarking
    cmd = [
        exe,
        "--rows", str(rows),
        "--cols", str(rows), # Square grid
        "--steps", str(steps),
        "--threads", str(threads),
        "--save-interval", "0", 
        "--no-progress"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse "Wall=X.XXXs" from stderr
        match = re.search(r"Wall=([0-9\.]+)s", result.stderr)
        if match:
            return float(match.group(1))
    except Exception as e:
        logger.error(f"Error running {exe}: {e}")
    return None

def generate_linspace(start, end, count):
    """Generates 'count' evenly spaced integers between start and end (inclusive)."""
    if count <= 0:
        logger.error("Count must be positive.")
        sys.exit(1)
    if count == 1:
        return [start]
    
    values = np.linspace(start, end, count)
    unique_values = sorted(list(set(np.round(values).astype(int))))
    return unique_values

def generate_float_range(start, end, step):
    """Generates an inclusive list of floats [start, end]."""
    if step <= 0:
        logger.error("Step size must be positive.")
        sys.exit(1)
    return list(np.arange(start, end + 1e-9, step))

def plot_line_metric(x_vals, y_series, x_label, y_label, title, filename, ideal_series=None):
    """Generic line plotter for individual files."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    for label, y_vals in y_series.items():
        plt.plot(x_vals, y_vals, marker='o', label=label)
    
    if ideal_series:
        plt.plot(x_vals, ideal_series, '--', color='gray', label='Ideal')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    
    if "Threads" in x_label:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(filename)
    plt.close()

def plot_heatmap(data_matrix, row_labels, col_labels, title, filename):
    """Generates a heatmap for Efficiency."""
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="RdYlGn", 
                     vmin=0, vmax=1.2, 
                     xticklabels=col_labels, yticklabels=row_labels)
    
    plt.title(title)
    plt.xlabel("Number of Threads")
    plt.ylabel("Grid Size (NxN)")
    ax.invert_yaxis() 
    plt.savefig(filename)
    plt.close()

def calculate_iso_efficiency(thread_counts, problem_sizes, results, serial_times, target_efficiencies):
    """
    Calculates the Problem Size N required to maintain a target Efficiency E 
    for each thread count. Uses linear interpolation.
    """
    iso_data = {target: [] for target in target_efficiencies}
    
    for T in thread_counts:
        if T == 1:
            for target in target_efficiencies:
                iso_data[target].append(np.nan)
            continue

        sizes = []
        effs = []
        for N in problem_sizes:
            t_ser = serial_times.get(N, 0)
            t_par = results[T].get(N, 0)
            if t_par > 0:
                speedup = t_ser / t_par
                efficiency = speedup / T
                sizes.append(N)
                effs.append(efficiency)
        
        if not sizes:
            for target in target_efficiencies:
                iso_data[target].append(np.nan)
            continue

        sorted_pairs = sorted(zip(effs, sizes))
        sorted_effs = [p[0] for p in sorted_pairs]
        sorted_sizes = [p[1] for p in sorted_pairs]

        for target in target_efficiencies:
            if target < min(sorted_effs) or target > max(sorted_effs):
                iso_data[target].append(np.nan) 
            else:
                required_N = np.interp(target, sorted_effs, sorted_sizes)
                iso_data[target].append(required_N)

    return iso_data

def main():
    parser = argparse.ArgumentParser(description="Benchmark Shallow Water Simulation.")
    
    # Grid Size Sweep Args
    parser.add_argument("--size-start", type=int, default=200, help="Start Grid Size (NxN)")
    parser.add_argument("--size-end", type=int, default=1000, help="End Grid Size (inclusive)")
    parser.add_argument("--size-count", type=int, default=5, help="Number of grid sizes to test")
    
    # Thread Sweep Args
    parser.add_argument("--thread-start", type=int, default=1, help="Start Thread Count")
    parser.add_argument("--thread-end", type=int, default=8, help="End Thread Count (inclusive)")
    parser.add_argument("--thread-count", type=int, default=8, help="Number of thread counts to test")
    
    # Iso-Efficiency Sweep Args
    parser.add_argument("--iso-start", type=float, default=0.5, help="Start Target Efficiency")
    parser.add_argument("--iso-end", type=float, default=0.9, help="End Target Efficiency")
    parser.add_argument("--iso-step", type=float, default=0.1, help="Target Efficiency Step")

    parser.add_argument("--steps", type=int, default=500, help="Fixed number of steps")
    
    args = parser.parse_args()

    # 1. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT_DIR, f"bench_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(run_dir, "experiment.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Output directory: {run_dir}")

    # 2. Generate Ranges (Using linspace count)
    problem_sizes = generate_linspace(args.size_start, args.size_end, args.size_count)
    thread_counts = generate_linspace(args.thread_start, args.thread_end, args.thread_count)
    iso_targets = generate_float_range(args.iso_start, args.iso_end, args.iso_step)
    iso_targets = [round(x, 2) for x in iso_targets]

    logger.info(f"Sweeping Grid Sizes: {problem_sizes}")
    logger.info(f"Sweeping Threads:    {thread_counts}")
    logger.info(f"Sweeping Iso-Eff Targets: {iso_targets}")

    # 3. Compile
    compile_simulation()
    
    exe_serial = os.path.join(C_CODE_DIR, "sw2d_serial")
    exe_omp = os.path.join(C_CODE_DIR, "sw2d_omp")
    exe_pthread = os.path.join(C_CODE_DIR, "sw2d_pthread")

    # Data Structures
    results_omp = {t: {} for t in thread_counts} 
    results_pthread = {t: {} for t in thread_counts}
    serial_times = {} 

    logger.info("Starting Benchmark Suite...")

    # --- NESTED SWEEP LOGIC ---
    for N in problem_sizes:
        logger.info(f"=== Testing Grid Size N={N} ===")
        
        # 4a. Run Serial Baseline (Once per N)
        logger.info(f"  Running Serial Baseline (N={N})...")
        t_ser = run_simulation(exe_serial, N, 1, args.steps)
        if t_ser is None: t_ser = 0.0
        serial_times[N] = t_ser
        logger.info(f"  -> Time: {t_ser:.4f}s")

        # 4b. Sweep OpenMP for this N
        logger.info(f"  Sweeping OpenMP (N={N})...")
        for T in thread_counts:
            t = run_simulation(exe_omp, N, T, args.steps)
            if t is None: t = 0.0
            results_omp[T][N] = t
            logger.info(f"    [OMP] Threads={T}: {t:.4f}s")

        # 4c. Sweep Pthreads for this N
        logger.info(f"  Sweeping Pthreads (N={N})...")
        for T in thread_counts:
            t = run_simulation(exe_pthread, N, T, args.steps)
            if t is None: t = 0.0
            results_pthread[T][N] = t
            logger.info(f"    [Pthread] Threads={T}: {t:.4f}s")

    # --- EXPORT TO CSV ---
    logger.info("Exporting results to CSV...")
    csv_file = os.path.join(run_dir, "benchmark_results.csv")
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Implementation", "Grid_Size", "Threads", "Time_Sec", "Speedup", "Efficiency"])
        
        # 1. Write Serial Data (Baseline)
        for N in problem_sizes:
            t = serial_times.get(N, 0.0)
            writer.writerow(["Serial", N, 1, f"{t:.6f}", "1.000000", "1.000000"])
            
        # 2. Write OpenMP Data
        for N in problem_sizes:
            t_ser = serial_times.get(N, 0.0)
            for T in thread_counts:
                t_par = results_omp[T].get(N, 0.0)
                if t_par > 0:
                    speedup = t_ser / t_par
                    efficiency = speedup / T
                else:
                    speedup = 0.0
                    efficiency = 0.0
                writer.writerow(["OpenMP", N, T, f"{t_par:.6f}", f"{speedup:.6f}", f"{efficiency:.6f}"])
                
        # 3. Write Pthreads Data
        for N in problem_sizes:
            t_ser = serial_times.get(N, 0.0)
            for T in thread_counts:
                t_par = results_pthread[T].get(N, 0.0)
                if t_par > 0:
                    speedup = t_ser / t_par
                    efficiency = speedup / T
                else:
                    speedup = 0.0
                    efficiency = 0.0
                writer.writerow(["Pthreads", N, T, f"{t_par:.6f}", f"{speedup:.6f}", f"{efficiency:.6f}"])

    logger.info(f"CSV Export Complete: {csv_file}")

    # --- PLOTTING LOGIC ---
    
    # Data Processing for Heatmaps
    eff_matrix_omp = np.zeros((len(problem_sizes), len(thread_counts)))
    eff_matrix_pthread = np.zeros((len(problem_sizes), len(thread_counts)))

    for r, N in enumerate(problem_sizes):
        t_ser = serial_times.get(N, 0)
        for c, T in enumerate(thread_counts):
            t_par_o = results_omp[T].get(N, 0)
            if t_par_o > 0 and T > 0:
                eff_matrix_omp[r, c] = (t_ser / t_par_o) / T
            
            t_par_p = results_pthread[T].get(N, 0)
            if t_par_p > 0 and T > 0:
                eff_matrix_pthread[r, c] = (t_ser / t_par_p) / T

    logger.info("Generating global heatmaps...")
    plot_heatmap(eff_matrix_omp, problem_sizes, thread_counts, 
                 "OpenMP Efficiency Heatmap", os.path.join(run_dir, "heatmap_efficiency_omp.png"))
    plot_heatmap(eff_matrix_pthread, problem_sizes, thread_counts, 
                 "Pthreads Efficiency Heatmap", os.path.join(run_dir, "heatmap_efficiency_pthread.png"))

    # --- C. MASTER COMPARISON PLOTS ---
    logger.info("Generating Master Comparison Plots...")
    prop_cycle = plt.rcParams['axes.prop_cycle']
    standard_colors = prop_cycle.by_key()['color']
    
    # 1. Combined Master Time
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        y_omp = [results_omp[T].get(N, 0) for T in thread_counts]
        y_pth = [results_pthread[T].get(N, 0) for T in thread_counts]
        plt.plot(thread_counts, y_omp, marker='o', linestyle='-', color=c, label=f"OMP (N={N})")
        plt.plot(thread_counts, y_pth, marker='x', linestyle='--', color=c, label=f"Pthread (N={N})")
    plt.xlabel("Number of Threads"); plt.ylabel("Time (s)")
    plt.title("Master Comparison: Time vs Threads (OMP vs Pthreads)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_comparison_time.png")); plt.close()

    # 2. Combined Master Speedup
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.plot(thread_counts, thread_counts, 'k:', linewidth=2, label="Ideal")
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        t_ser = serial_times.get(N, 0)
        s_omp = []
        s_pth = []
        for T in thread_counts:
            t = results_omp[T].get(N, 0)
            s_omp.append(t_ser/t if t > 0 else 0)
            t = results_pthread[T].get(N, 0)
            s_pth.append(t_ser/t if t > 0 else 0)
        plt.plot(thread_counts, s_omp, marker='o', linestyle='-', color=c, label=f"OMP (N={N})")
        plt.plot(thread_counts, s_pth, marker='x', linestyle='--', color=c, label=f"Pthread (N={N})")
    plt.xlabel("Number of Threads"); plt.ylabel("Speedup")
    plt.title("Master Comparison: Speedup vs Threads (OMP vs Pthreads)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_comparison_speedup.png")); plt.close()

    # 3. Combined Master Efficiency
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.axhline(y=1.0, color='gray', linestyle=':', label="Ideal")
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        t_ser = serial_times.get(N, 0)
        e_omp = []
        e_pth = []
        for T in thread_counts:
            t = results_omp[T].get(N, 0)
            s = t_ser/t if t > 0 else 0
            e_omp.append(s/T if T > 0 else 0)
            t = results_pthread[T].get(N, 0)
            s = t_ser/t if t > 0 else 0
            e_pth.append(s/T if T > 0 else 0)
        plt.plot(thread_counts, e_omp, marker='o', linestyle='-', color=c, label=f"OMP (N={N})")
        plt.plot(thread_counts, e_pth, marker='x', linestyle='--', color=c, label=f"Pthread (N={N})")
    plt.xlabel("Number of Threads"); plt.ylabel("Efficiency")
    plt.title("Master Comparison: Efficiency vs Threads (OMP vs Pthreads)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.ylim(0, 1.2); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_comparison_efficiency.png")); plt.close()

    # --- D. INDIVIDUAL MASTER PLOTS (OMP ONLY & PTHREADS ONLY) ---
    logger.info("Generating Individual Master Plots...")

    # 4. OpenMP Master Time (NEW)
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        y_omp = [results_omp[T].get(N, 0) for T in thread_counts]
        plt.plot(thread_counts, y_omp, marker='o', linestyle='-', color=c, label=f"N={N}")
    plt.xlabel("Number of Threads"); plt.ylabel("Time (s)")
    plt.title("OpenMP Master Time Scaling")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_omp_time.png")); plt.close()

    # 5. Pthreads Master Time (NEW)
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        y_pth = [results_pthread[T].get(N, 0) for T in thread_counts]
        plt.plot(thread_counts, y_pth, marker='x', linestyle='--', color=c, label=f"N={N}")
    plt.xlabel("Number of Threads"); plt.ylabel("Time (s)")
    plt.title("Pthreads Master Time Scaling")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_pthread_time.png")); plt.close()

    # 6. OpenMP Master Speedup
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.plot(thread_counts, thread_counts, 'k:', linewidth=2, label="Ideal")
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        t_ser = serial_times.get(N, 0)
        s_omp = []
        for T in thread_counts:
            t = results_omp[T].get(N, 0)
            s_omp.append(t_ser/t if t > 0 else 0)
        plt.plot(thread_counts, s_omp, marker='o', linestyle='-', color=c, label=f"N={N}")
    plt.xlabel("Number of Threads"); plt.ylabel("Speedup")
    plt.title("OpenMP Master Speedup Scaling")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_omp_speedup.png")); plt.close()

    # 7. Pthreads Master Speedup
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.plot(thread_counts, thread_counts, 'k:', linewidth=2, label="Ideal")
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        t_ser = serial_times.get(N, 0)
        s_pth = []
        for T in thread_counts:
            t = results_pthread[T].get(N, 0)
            s_pth.append(t_ser/t if t > 0 else 0)
        plt.plot(thread_counts, s_pth, marker='x', linestyle='--', color=c, label=f"N={N}")
    plt.xlabel("Number of Threads"); plt.ylabel("Speedup")
    plt.title("Pthreads Master Speedup Scaling")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_pthread_speedup.png")); plt.close()

    # 8. OpenMP Master Efficiency
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.axhline(y=1.0, color='gray', linestyle=':', label="Ideal")
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        t_ser = serial_times.get(N, 0)
        e_omp = []
        for T in thread_counts:
            t = results_omp[T].get(N, 0)
            s = t_ser/t if t > 0 else 0
            e_omp.append(s/T if T > 0 else 0)
        plt.plot(thread_counts, e_omp, marker='o', linestyle='-', color=c, label=f"N={N}")
    plt.xlabel("Number of Threads"); plt.ylabel("Efficiency")
    plt.title("OpenMP Master Efficiency Scaling")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.ylim(0, 1.2); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_omp_efficiency.png")); plt.close()

    # 9. Pthreads Master Efficiency
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.axhline(y=1.0, color='gray', linestyle=':', label="Ideal")
    for i, N in enumerate(problem_sizes):
        c = standard_colors[i % len(standard_colors)]
        t_ser = serial_times.get(N, 0)
        e_pth = []
        for T in thread_counts:
            t = results_pthread[T].get(N, 0)
            s = t_ser/t if t > 0 else 0
            e_pth.append(s/T if T > 0 else 0)
        plt.plot(thread_counts, e_pth, marker='x', linestyle='--', color=c, label=f"N={N}")
    plt.xlabel("Number of Threads"); plt.ylabel("Efficiency")
    plt.title("Pthreads Master Efficiency Scaling")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.ylim(0, 1.2); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "master_pthread_efficiency.png")); plt.close()

    # --- E. PER-SIZE PLOTS ---
    logger.info("Generating per-problem-size results...")
    for N in problem_sizes:
        size_dir = os.path.join(run_dir, f"N_{N}")
        os.makedirs(size_dir, exist_ok=True)
        
        t_ser_fixed = serial_times.get(N, 0)
        times_omp = [results_omp[T].get(N, 0) for T in thread_counts]
        times_pthread = [results_pthread[T].get(N, 0) for T in thread_counts]
        
        speedup_omp = [t_ser_fixed/t if t>0 else 0 for t in times_omp]
        speedup_pthread = [t_ser_fixed/t if t>0 else 0 for t in times_pthread]
        
        eff_omp_line = [s/T if T>0 else 0 for s, T in zip(speedup_omp, thread_counts)]
        eff_pthread_line = [s/T if T>0 else 0 for s, T in zip(speedup_pthread, thread_counts)]

        plot_line_metric(thread_counts, {"OpenMP": times_omp}, "Threads", "Time (s)", 
                         f"OpenMP Execution Time (N={N})", os.path.join(size_dir, "individual_time_omp.png"))
        plot_line_metric(thread_counts, {"Pthreads": times_pthread}, "Threads", "Time (s)", 
                         f"Pthreads Execution Time (N={N})", os.path.join(size_dir, "individual_time_pthread.png"))

        plot_line_metric(thread_counts, {"OpenMP": speedup_omp}, "Threads", "Speedup", 
                         f"OpenMP Speedup (N={N})", os.path.join(size_dir, "individual_speedup_omp.png"), 
                         ideal_series=thread_counts)
        plot_line_metric(thread_counts, {"Pthreads": speedup_pthread}, "Threads", "Speedup", 
                         f"Pthreads Speedup (N={N})", os.path.join(size_dir, "individual_speedup_pthread.png"), 
                         ideal_series=thread_counts)

        plot_line_metric(thread_counts, {"OpenMP": eff_omp_line}, "Threads", "Efficiency", 
                         f"OpenMP Efficiency (N={N})", os.path.join(size_dir, "individual_efficiency_omp.png"),
                         ideal_series=[1.0]*len(thread_counts))
        plot_line_metric(thread_counts, {"Pthreads": eff_pthread_line}, "Threads", "Efficiency", 
                         f"Pthreads Efficiency (N={N})", os.path.join(size_dir, "individual_efficiency_pthread.png"),
                         ideal_series=[1.0]*len(thread_counts))

        plot_line_metric(thread_counts, {"OpenMP": times_omp, "Pthreads": times_pthread}, 
                         "Threads", "Time (s)", f"Execution Time Comparison (N={N})", 
                         os.path.join(size_dir, "comparison_time.png"))

        plot_line_metric(thread_counts, {"OpenMP": speedup_omp, "Pthreads": speedup_pthread}, 
                         "Threads", "Speedup", f"Speedup Comparison (N={N})", 
                         os.path.join(size_dir, "comparison_speedup.png"), 
                         ideal_series=thread_counts)

        plot_line_metric(thread_counts, {"OpenMP": eff_omp_line, "Pthreads": eff_pthread_line}, 
                         "Threads", "Efficiency", f"Efficiency Comparison (N={N})", 
                         os.path.join(size_dir, "comparison_efficiency.png"),
                         ideal_series=[1.0]*len(thread_counts))

    # --- F. ISO-EFFICIENCY PLOTS ---
    logger.info("Generating Iso-Efficiency Function Plots...")
    iso_dir = os.path.join(run_dir, "iso_efficiency_lines")
    os.makedirs(iso_dir, exist_ok=True)

    iso_data_omp = calculate_iso_efficiency(thread_counts, problem_sizes, results_omp, serial_times, iso_targets)
    iso_data_pthread = calculate_iso_efficiency(thread_counts, problem_sizes, results_pthread, serial_times, iso_targets)

    plot_line_metric(thread_counts, {f"E={t:.2f}": val for t, val in iso_data_omp.items()}, 
                     "Threads", "Required Grid Size (N)", 
                     "OpenMP Iso-Efficiency Function", 
                     os.path.join(iso_dir, "iso_efficiency_func_omp.png"))

    plot_line_metric(thread_counts, {f"E={t:.2f}": val for t, val in iso_data_pthread.items()}, 
                     "Threads", "Required Grid Size (N)", 
                     "Pthreads Iso-Efficiency Function", 
                     os.path.join(iso_dir, "iso_efficiency_func_pthread.png"))
    
    subset_targets = iso_targets[::max(1, len(iso_targets)//3)]
    
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for i, target in enumerate(subset_targets):
        c = standard_colors[i % len(standard_colors)]
        if target in iso_data_omp:
            plt.plot(thread_counts, iso_data_omp[target], 'o-', color=c, label=f"OMP (E={target:.2f})")
        if target in iso_data_pthread:
            plt.plot(thread_counts, iso_data_pthread[target], 'x--', color=c, label=f"Pthread (E={target:.2f})")
            
    plt.title("Iso-Efficiency Comparison (Selected Targets)")
    plt.xlabel("Threads")
    plt.ylabel("Required Grid Size (N)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(iso_dir, "iso_efficiency_comparison_sweep.png"))
    plt.close()

    # Individual Iso Plots
    logger.info("Generating individual plots for each Iso-Efficiency target...")
    for target in iso_targets:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        has_data = False
        
        if target in iso_data_omp:
            plt.plot(thread_counts, iso_data_omp[target], 'o-', label=f"OpenMP")
            has_data = True
            
        if target in iso_data_pthread:
            plt.plot(thread_counts, iso_data_pthread[target], 'x--', label=f"Pthreads")
            has_data = True
            
        if has_data:
            plt.title(f"Iso-Efficiency Function for E = {target:.2f}")
            plt.xlabel("Number of Threads")
            plt.ylabel("Required Grid Size (N)")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            filename = f"iso_efficiency_E_{target:.2f}.png"
            plt.savefig(os.path.join(iso_dir, filename))
        
        plt.close()

    logger.info(f"Benchmarks complete. Results saved in {run_dir}")

if __name__ == "__main__":
    main()