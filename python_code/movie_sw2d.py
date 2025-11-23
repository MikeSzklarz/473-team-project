import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import os
import sys
import logging

# Configuration
C_CODE_DIR = "../c_code"

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

def generate_default_movie(output_file, rows, cols, steps, save_int):
    """Runs the serial simulation to generate a movie file."""
    exe_serial = os.path.join(C_CODE_DIR, "sw2d_serial")
    
    if not os.path.exists(exe_serial):
        logger.error("Executable not found even after compilation step.")
        sys.exit(1)
        
    logger.info(f"Generating data for movie ({output_file})...")
    logger.info(f"(Running {rows}x{cols} grid for {steps} steps, saving every {save_int} steps)")
    
    cmd = [
        exe_serial,
        "--rows", str(rows),
        "--cols", str(cols),
        "--steps", str(steps),
        "--save-interval", str(save_int),
        "--out", output_file
    ]
    
    try:
        subprocess.check_call(cmd)
        logger.info("Data generation complete.")
    except subprocess.CalledProcessError:
        logger.error("Failed to run simulation.")
        sys.exit(1)

def read_frames(filepath):
    """Reads the binary file format."""
    with open(filepath, 'rb') as f:
        header = f.read(68)
        if len(header) < 68: return None, None, None, None
        magic, ver, flags, rows, cols, nframes, sav_int, dx, dy, dt, g, h0 = struct.unpack('<4sIIiiiiddddd', header)
        
        frames = []
        bytes_per_layer = rows * cols * 8
        bytes_per_frame = 3 * bytes_per_layer
        
        for _ in range(nframes):
            data = f.read(bytes_per_frame)
            if len(data) < bytes_per_frame: break
            # We only care about 'h' (the first layer) for the movie
            h_data = data[:bytes_per_layer]
            h = np.frombuffer(h_data, dtype=np.float64).reshape((rows, cols))
            frames.append(h)
            
    return frames, dt * sav_int, rows, cols

def main():
    parser = argparse.ArgumentParser(description="Visualize Shallow Water Simulation (2D & 3D).")
    parser.add_argument("--file", type=str, default="movie.bin", help="Input binary file (default: movie.bin)")
    parser.add_argument("--force-regen", action="store_true", help="Force regeneration of bin file")
    parser.add_argument("--rows", type=int, default=200, help="Regen: Grid rows (default: 200)")
    parser.add_argument("--cols", type=int, default=200, help="Regen: Grid cols (default: 200)")
    parser.add_argument("--steps", type=int, default=1000, help="Regen: Simulation steps (default: 1000)")
    parser.add_argument("--save-interval", type=int, default=10, help="Regen: Save interval (default: 10)")
    
    # Updated Argument: nargs='?' allows it to be used as a flag (uses const) or with a value
    parser.add_argument("--save-mp4", nargs='?', const="simulation.mp4", type=str, default=None, 
                        help="Optional: Save animation to MP4 file. Defaults to 'simulation.mp4' if flag is present but no filename given.")
    
    parser.add_argument("--fps", type=int, default=20, help="Frames Per Second for playback (default: 20)")
    
    args = parser.parse_args()

    # 1. Compile Code (Automated)
    compile_simulation()

    # 2. Check for File / Auto-Generate
    if args.force_regen or not os.path.exists(args.file):
        if not os.path.exists(args.file):
            logger.warning(f"Input file '{args.file}' not found.")
        generate_default_movie(args.file, args.rows, args.cols, args.steps, args.save_interval)
    
    # 3. Read Data
    logger.info(f"Loading frames from {args.file}...")
    try:
        frames, dt_step, rows, cols = read_frames(args.file)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return

    if not frames:
        logger.error("No frames found in file.")
        return

    logger.info(f"Loaded {len(frames)} frames. Preparing visualization at {args.fps} FPS...")

    # 4. Setup Visualization
    fig = plt.figure(figsize=(14, 6))
    
    # 2D Subplot (Left)
    ax2d = fig.add_subplot(1, 2, 1)
    # Use fixed vmin/vmax to visualize wave heights accurately
    im = ax2d.imshow(frames[0], cmap='seismic', vmin=-0.5, vmax=0.5, origin='lower')
    fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04, label="Height (h)")
    ax2d.set_title("2D Heatmap")
    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Y")

    # 3D Subplot (Right)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Performance optimization: Downsample stride for 3D wireframe/surface
    # Matplotlib 3D is slow. To get smooth 30+ FPS, we must limit polygon count.
    # Target: ~40x40 grid points regardless of actual simulation resolution.
    target_res = 40
    stride_row = max(1, rows // target_res)
    stride_col = max(1, cols // target_res)

    # Initial 3D plot
    def plot_3d_frame(frame):
        ax3d.clear()
        ax3d.set_zlim(-1.0, 1.0) # Fixed Z-axis to prevent jitter
        ax3d.set_title("3D Surface")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Height")
        
        # Plot surface with downsampling (rstride/cstride) for performance
        s = ax3d.plot_surface(X, Y, frame, cmap='seismic', vmin=-0.5, vmax=0.5, 
                              rstride=stride_row, cstride=stride_col, 
                              linewidth=0, antialiased=False)
        return s

    plot_3d_frame(frames[0])
    
    # Title Text
    main_title = fig.suptitle(f"Time: 0.000s", fontsize=16)

    def animate(i):
        # Update Title
        main_title.set_text(f"Time: {i * dt_step:.3f}s")
        
        # Update 2D
        im.set_data(frames[i])
        
        # Update 3D
        plot_3d_frame(frames[i])
        
        return im,

    # Create Animation
    # Calculate interval from FPS (1000ms / fps)
    interval_ms = 1000 / args.fps
    
    logger.info(f"Starting animation ({args.fps} FPS, Interval: {interval_ms:.1f}ms)...")
    
    # Note: Blit=False is generally required for 3D animations in Matplotlib to render correctly,
    # which is why aggressive downsampling (stride) is necessary for speed.
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=interval_ms, blit=False)

    if args.save_mp4:
        logger.info(f"Saving animation to {args.save_mp4} (this may take a while)...")
        # Use higher bitrate for better quality output
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2000)
        try:
            anim.save(args.save_mp4, writer=writer)
            logger.info(f"Saved {args.save_mp4} successfully.")
        except Exception as e:
            logger.error(f"Failed to save video (ffmpeg installed?): {e}")
    else:
        plt.show()

if __name__ == "__main__":
    main()