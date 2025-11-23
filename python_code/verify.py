import struct
import numpy as np
import subprocess
import os
import sys
import argparse
import logging
from datetime import datetime

# Configuration
C_CODE_DIR = "../c_code"
VERIFY_ROOT_DIR = "../verify_runs"

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def compile_simulation():
    """Compiles the C code using the Makefile in c_code directory."""
    logger.info("Compiling C simulation code...")
    try:
        # Run make in the c_code directory
        subprocess.check_call(["make", "-C", C_CODE_DIR], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Compilation successful.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Compilation failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Could not find 'make' or the directory '{C_CODE_DIR}'.")
        sys.exit(1)

def read_final_frame(filepath):
    """Reads only the final H frame from the binary."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(68)
            if len(header) < 68: return None
            magic, ver, flags, rows, cols, nframes, sav_int, dx, dy, dt, g, h0 = struct.unpack('<4sIIiiiiddddd', header)
            
            frame_size = 3 * rows * cols * 8
            if nframes == 0: return None
            
            f.seek(68 + (nframes - 1) * frame_size)
            h_data = f.read(rows * cols * 8)
            h_arr = np.frombuffer(h_data, dtype=np.float64).reshape((rows, cols))
            return h_arr
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None

def generate_range(start, end, step):
    """Generates an inclusive list of integers [start, end]."""
    if step <= 0:
        logger.error("Step size must be positive.")
        sys.exit(1)
    # Use end + 1 to make it inclusive
    return list(range(start, end + 1, step))

def run_verification(args):
    # 1. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(VERIFY_ROOT_DIR, f"verify_sweep_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Add file handler to logger
    file_handler = logging.FileHandler(os.path.join(run_dir, "verification_sweep.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Output directory: {run_dir}")
    
    # 2. Generate Ranges
    grid_sizes = generate_range(args.size_start, args.size_end, args.size_step)
    step_counts = generate_range(args.step_start, args.step_end, args.step_step)
    thread_counts = generate_range(args.thread_start, args.thread_end, args.thread_step)

    logger.info(f"Sweeping Grid Sizes: {grid_sizes}")
    logger.info(f"Sweeping Steps:      {step_counts}")
    logger.info(f"Sweeping Threads:    {thread_counts}")

    # 3. Compile Code
    compile_simulation()

    # Define paths for executables
    exe_serial = os.path.join(C_CODE_DIR, "sw2d_serial")
    exe_omp = os.path.join(C_CODE_DIR, "sw2d_omp")
    exe_pthread = os.path.join(C_CODE_DIR, "sw2d_pthread")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    # 4. Sweep Nested Loops
    # Loop 1: Grid Sizes
    for N in grid_sizes:
        # Loop 2: Step Counts
        for steps in step_counts:
            logger.info(f"=== Testing Config: Grid {N}x{N}, Steps {steps} ===")
            
            # 4a. Run Serial Baseline (Only once per N/steps config)
            ref_file = os.path.join(run_dir, f"ref_{N}_{steps}.bin")
            logger.info("  Running Serial Reference...")
            
            cmd_serial = [
                exe_serial, 
                "--rows", str(N), "--cols", str(N), 
                "--steps", str(steps), 
                "--save-interval", str(steps), # Optimization: Only save the last frame
                "--out", ref_file, 
                "--no-progress"
            ]
            
            try:
                subprocess.run(cmd_serial, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                logger.error("  Serial execution failed. Skipping this configuration.")
                continue
                
            ref_data = read_final_frame(ref_file)
            if ref_data is None:
                logger.error("  Failed to read reference data. Skipping.")
                continue

            # Loop 3: Thread Counts
            for T in thread_counts:
                # ==========================================
                # Test 1: OpenMP
                # ==========================================
                file_omp = os.path.join(run_dir, f"test_omp_{N}_{steps}_{T}t.bin")
                logger.info(f"  [OMP] Comparing with {T} threads...")
                
                cmd_omp = [
                    exe_omp, 
                    "--rows", str(N), "--cols", str(N), 
                    "--steps", str(steps), 
                    "--save-interval", str(steps), 
                    "--out", file_omp, 
                    "--threads", str(T), 
                    "--no-progress"
                ]
                
                try:
                    subprocess.run(cmd_omp, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    test_data_omp = read_final_frame(file_omp)
                    
                    if test_data_omp is None:
                         raise ValueError("No data read")

                    diff_omp = np.max(np.abs(ref_data - test_data_omp))
                    is_match_omp = np.allclose(ref_data, test_data_omp, atol=1e-10)
                    
                    total_tests += 1
                    if is_match_omp:
                        logger.info(f"    PASS [OMP] (Threads={T}): Max Diff = {diff_omp:.5e}")
                        passed_tests += 1
                    else:
                        logger.error(f"    FAIL [OMP] (Threads={T}): Max Diff = {diff_omp:.5e}")
                        failed_tests += 1
                        
                except (subprocess.CalledProcessError, ValueError):
                    logger.error(f"    FAIL [OMP] (Threads={T}): Execution or Read Error")
                    failed_tests += 1
                    total_tests += 1

                # ==========================================
                # Test 2: Pthreads
                # ==========================================
                file_pthread = os.path.join(run_dir, f"test_pthread_{N}_{steps}_{T}t.bin")
                logger.info(f"  [Pthread] Comparing with {T} threads...")
                
                cmd_pthread = [
                    exe_pthread, 
                    "--rows", str(N), "--cols", str(N), 
                    "--steps", str(steps), 
                    "--save-interval", str(steps), 
                    "--out", file_pthread, 
                    "--threads", str(T), 
                    "--no-progress"
                ]
                
                try:
                    subprocess.run(cmd_pthread, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    test_data_pthread = read_final_frame(file_pthread)
                    
                    if test_data_pthread is None:
                        raise ValueError("No data read")

                    diff_pthread = np.max(np.abs(ref_data - test_data_pthread))
                    is_match_pthread = np.allclose(ref_data, test_data_pthread, atol=1e-10)
                    
                    total_tests += 1
                    if is_match_pthread:
                        logger.info(f"    PASS [Pthread] (Threads={T}): Max Diff = {diff_pthread:.5e}")
                        passed_tests += 1
                    else:
                        logger.error(f"    FAIL [Pthread] (Threads={T}): Max Diff = {diff_pthread:.5e}")
                        failed_tests += 1

                except (subprocess.CalledProcessError, ValueError):
                    logger.error(f"    FAIL [Pthread] (Threads={T}): Execution or Read Error")
                    failed_tests += 1
                    total_tests += 1

    # 5. Final Summary
    logger.info("-" * 40)
    logger.info("VERIFICATION SWEEP SUMMARY")
    logger.info("-" * 40)
    logger.info(f"Total Comparisons: {total_tests}")
    logger.info(f"Passed:            {passed_tests}")
    logger.info(f"Failed:            {failed_tests}")
    logger.info("-" * 40)
    
    if failed_tests > 0:
        logger.warning("Some tests failed. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All tests passed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify OpenMP and Pthreads implementations with parameter sweep.")
    
    # Grid Size Sweep Args
    parser.add_argument("--size-start", type=int, default=100, help="Start Grid Size (NxN) (default: 100)")
    parser.add_argument("--size-end", type=int, default=400, help="End Grid Size (inclusive) (default: 400)")
    parser.add_argument("--size-step", type=int, default=100, help="Grid Size Step (default: 100)")
    
    # Steps Sweep Args
    parser.add_argument("--step-start", type=int, default=100, help="Start Step Count (default: 100)")
    parser.add_argument("--step-end", type=int, default=500, help="End Step Count (inclusive) (default: 500)")
    parser.add_argument("--step-step", type=int, default=400, help="Step Count Step (default: 400)")
    
    # Thread Sweep Args
    parser.add_argument("--thread-start", type=int, default=1, help="Start Thread Count (default: 1)")
    parser.add_argument("--thread-end", type=int, default=8, help="End Thread Count (inclusive) (default: 8)")
    parser.add_argument("--thread-step", type=int, default=1, help="Thread Count Step (default: 1)")
    
    args = parser.parse_args()
    run_verification(args)