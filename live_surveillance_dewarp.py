import os
import sys
import argparse
import glob
import cv2
import numpy as np

# Global variables for state management (used in interactive mode)
mode = 'perspective'  # 'panorama', 'double_panorama', 'perspective', 'quad'
yaw = 0.0
pitch = 60.0
roll = 0.0
fov_out = 100.0

# Lens calibration parameters
Cx = -1.0
Cy = -1.0
R_max = -1.0
fov_in = 180.0

# Interactive mouse state
is_dragging = False
start_x, start_y = 0, 0
start_yaw, start_pitch = 0.0, 0.0
map_needs_update = True
width_out = 1280
height_out = 720


def make_panorama_map(W_in, H_in, W_out, H_out, cx, cy, r_max):
    """
    dewarps the circular fisheye to a 360-degree cylindrical panorama.
    """
    u = np.arange(W_out, dtype=np.float32)
    v = np.arange(H_out, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)

    # Map column to azimuth angle theta [0, 2*pi]. Offset by -pi/2 so top is centered.
    theta = 2.0 * np.pi * u_grid / W_out - np.pi / 2.0
    # Map row to radial distance (outer edge is top of image, center is bottom of image)
    r = r_max * (1.0 - v_grid / H_out)

    map_x = cx + r * np.cos(theta)
    map_y = cy + r * np.sin(theta)

    # Set out-of-bounds to black
    mask = r > r_max
    map_x[mask] = -1
    map_y[mask] = -1

    return map_x.astype(np.float32), map_y.astype(np.float32)


def make_double_panorama_map(W_in, H_in, W_out, H_out, cx, cy, r_max):
    """
    dewarps the circular fisheye into two 180-degree panoramas stacked vertically
    (front half on top, back half on bottom).
    """
    half_H = H_out // 2
    u = np.arange(W_out, dtype=np.float32)

    # Top half: front 180 degrees (-pi/2 to pi/2)
    v_top = np.arange(half_H, dtype=np.float32)
    u_top, v_top_grid = np.meshgrid(u, v_top)
    theta_top = np.pi * u_top / W_out - np.pi / 2.0
    r_top = r_max * (1.0 - v_top_grid / half_H)
    map_x_top = cx + r_top * np.cos(theta_top)
    map_y_top = cy + r_top * np.sin(theta_top)

    # Bottom half: back 180 degrees (pi/2 to 3pi/2)
    v_bot = np.arange(half_H, dtype=np.float32)
    u_bot, v_bot_grid = np.meshgrid(u, v_bot)
    theta_bot = np.pi * u_bot / W_out + np.pi / 2.0
    r_bot = r_max * (1.0 - v_bot_grid / half_H)
    map_x_bot = cx + r_bot * np.cos(theta_bot)
    map_y_bot = cy + r_bot * np.sin(theta_bot)

    map_x = np.vstack([map_x_top, map_x_bot])
    map_y = np.vstack([map_y_top, map_y_bot])

    return map_x.astype(np.float32), map_y.astype(np.float32)


def make_perspective_map(W_in, H_in, W_out, H_out, cx, cy, r_max, f_in, y_deg, p_deg, r_deg, f_out_deg):
    """
    dewarps circular fisheye to perspective rectilinear view (virtual PTZ camera).
    """
    y = np.radians(y_deg)
    p = np.radians(p_deg)
    r_rad = np.radians(r_deg)

    # Rotation matrix: yaw -> pitch -> roll
    R_yaw = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p), np.cos(p)]
    ], dtype=np.float32)

    R_roll = np.array([
        [np.cos(r_rad), -np.sin(r_rad), 0],
        [np.sin(r_rad), np.cos(r_rad), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R = R_yaw @ R_pitch @ R_roll

    u = np.arange(W_out, dtype=np.float32)
    v = np.arange(H_out, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)

    # Target virtual camera focal length
    f_out = (W_out / 2.0) / np.tan(np.radians(f_out_deg / 2.0))

    # Ray vectors from virtual camera
    x_c = (u_grid - W_out / 2.0) / f_out
    y_c = (v_grid - H_out / 2.0) / f_out
    z_c = np.ones_like(x_c)

    pts = np.vstack([x_c.ravel(), y_c.ravel(), z_c.ravel()])
    pts_rot = R @ pts

    X = pts_rot[0, :]
    Y = pts_rot[1, :]
    Z = pts_rot[2, :]

    norm = np.sqrt(X*X + Y*Y + Z*Z)
    norm[norm == 0] = 1e-6
    phi = np.arccos(Z / norm)
    theta = np.arctan2(Y, X)

    # Map zenith angle to radial distance on sensor (equidistant model)
    phi_max = np.radians(f_in / 2.0)
    r = r_max * (phi / phi_max)

    map_x = cx + r * np.cos(theta)
    map_y = cy + r * np.sin(theta)

    # Mask out-of-lens rays
    invalid = (phi > phi_max) | (phi < 0)
    map_x[invalid] = -1
    map_y[invalid] = -1

    map_x = map_x.reshape((H_out, W_out))
    map_y = map_y.reshape((H_out, W_out))

    return map_x.astype(np.float32), map_y.astype(np.float32)


def make_quad_map(W_in, H_in, W_out, H_out, cx, cy, r_max, f_in, f_out_deg):
    """
    dewarps circular fisheye into a 2x2 grid representing 4 perspective cameras (N, E, S, W).
    """
    half_W = W_out // 2
    half_H = H_out // 2

    map_x1, map_y1 = make_perspective_map(W_in, H_in, half_W, half_H, cx, cy, r_max, f_in, 0.0, 55.0, 0.0, f_out_deg)
    map_x2, map_y2 = make_perspective_map(W_in, H_in, half_W, half_H, cx, cy, r_max, f_in, 90.0, 55.0, 0.0, f_out_deg)
    map_x3, map_y3 = make_perspective_map(W_in, H_in, half_W, half_H, cx, cy, r_max, f_in, 180.0, 55.0, 0.0, f_out_deg)
    map_x4, map_y4 = make_perspective_map(W_in, H_in, half_W, half_H, cx, cy, r_max, f_in, 270.0, 55.0, 0.0, f_out_deg)

    map_x = np.vstack([
        np.hstack([map_x1, map_x2]),
        np.hstack([map_x3, map_x4])
    ])
    map_y = np.vstack([
        np.hstack([map_y1, map_y2]),
        np.hstack([map_y3, map_y4])
    ])

    return map_x.astype(np.float32), map_y.astype(np.float32)


def mouse_callback(event, x, y, flags, param):
    global is_dragging, start_x, start_y, start_yaw, start_pitch, yaw, pitch, map_needs_update
    if mode != 'perspective':
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        start_x, start_y = x, y
        start_yaw, start_pitch = yaw, pitch
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            dx = x - start_x
            dy = y - start_y
            sensitivity = 120.0 / width_out  # Dragging screen width rotates 120 degrees
            yaw = (start_yaw - dx * sensitivity) % 360.0
            pitch = np.clip(start_pitch + dy * sensitivity, 0.0, 90.0)
            map_needs_update = True
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False


def process_video(video_path, args, is_headless=False, default_mode=None):
    global mode, yaw, pitch, roll, fov_out, Cx, Cy, R_max, fov_in, map_needs_update, width_out, height_out

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Video parameters
    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0
    delay = int(1000 / fps)

    # Set up defaults for center and radius
    if args.cx < 0:
        Cx = W_in / 2.0
    else:
        Cx = args.cx

    if args.cy < 0:
        Cy = H_in / 2.0
    else:
        Cy = args.cy

    if args.radius < 0:
        R_max = min(W_in, H_in) / 2.0
    else:
        R_max = args.radius

    fov_in = args.fov_in
    width_out = args.width
    height_out = args.height

    if default_mode:
        mode = default_mode
    else:
        mode = args.mode

    # Setup Video Writer if output is required
    out_video = None
    if args.output:
        # Determine output filename
        if os.path.isdir(args.output) or not args.output.endswith('.mp4'):
            os.makedirs(args.output, exist_ok=True)
            base = os.path.basename(video_path)
            name, _ = os.path.splitext(base)
            out_path = os.path.join(args.output, f"{name}_defished_{mode}.mp4")
        else:
            out_path = args.output

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (width_out, height_out))
        print(f"Saving dewarped output to: {out_path}")

    # Headless Batch Processing Mode
    if is_headless:
        print(f"Batch processing {video_path} in '{mode}' mode using FFmpeg...")
        
        # Release resources initialized for OpenCV loop
        cap.release()
        if out_video is not None:
            out_video.release()

        # Map yaw from 0..360 to FFmpeg's [-180, 180] range
        yaw_ff = yaw % 360.0
        if yaw_ff > 180.0:
            yaw_ff -= 360.0

        # Construct FFmpeg filters
        half_H = height_out // 2
        half_W = width_out // 2

        if mode == 'panorama':
            vf = f"v360=input=fisheye:output=equirect:ih_fov={fov_in}:iv_fov={fov_in}:pitch=-90:w={width_out}:h={height_out}"
        elif mode == 'double_panorama':
            # Stack front and back 180 panoramas
            vf = (
                f"[0:v]v360=input=fisheye:output=equirect:ih_fov={fov_in}:iv_fov={fov_in}:pitch=-90:yaw=0:h_fov=180:v_fov=90:w={width_out}:h={half_H}[top]; "
                f"[0:v]v360=input=fisheye:output=equirect:ih_fov={fov_in}:iv_fov={fov_in}:pitch=-90:yaw=-180:h_fov=180:v_fov=90:w={width_out}:h={half_H}[bottom]; "
                f"[top][bottom]vstack=inputs=2[out]"
            )
        elif mode == 'perspective':
            vf = f"v360=input=fisheye:output=flat:ih_fov={fov_in}:iv_fov={fov_in}:pitch={pitch}:yaw={yaw_ff}:roll={roll}:h_fov={fov_out}:v_fov={fov_out*9/16}:w={width_out}:h={height_out}"
        elif mode == 'quad':
            vf = (
                f"[0:v]v360=input=fisheye:output=flat:ih_fov={fov_in}:iv_fov={fov_in}:pitch=55:yaw=0:h_fov=90:v_fov=90:w={half_W}:h={half_H}[n]; "
                f"[0:v]v360=input=fisheye:output=flat:ih_fov={fov_in}:iv_fov={fov_in}:pitch=55:yaw=90:h_fov=90:v_fov=90:w={half_W}:h={half_H}[e]; "
                f"[0:v]v360=input=fisheye:output=flat:ih_fov={fov_in}:iv_fov={fov_in}:pitch=55:yaw=180:h_fov=90:v_fov=90:w={half_W}:h={half_H}[s]; "
                f"[0:v]v360=input=fisheye:output=flat:ih_fov={fov_in}:iv_fov={fov_in}:pitch=55:yaw=-90:h_fov=90:v_fov=90:w={half_W}:h={half_H}[w]; "
                f"[n][e]hstack=inputs=2[top]; "
                f"[s][w]hstack=inputs=2[bottom]; "
                f"[top][bottom]vstack=inputs=2[out]"
            )

        cmd = ["ffmpeg", "-y", "-i", video_path]
        if mode in ['double_panorama', 'quad']:
            cmd.extend(["-filter_complex", vf, "-map", "[out]"])
        else:
            cmd.extend(["-vf", vf])

        cmd.extend([
            "-c:v", "mpeg4",
            "-qscale:v", "3",
            "-pix_fmt", "yuv420p",
            out_path
        ])

        print(f"Executing command: {' '.join(cmd)}")
        
        import subprocess
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {video_path} -> {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error: FFmpeg failed with exit code {e.returncode}")
        return

    # Interactive GUI mode
    window_name = "Surveillance Dewarp - Interactive PTZ"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    map_x, map_y = None, None
    map_needs_update = True

    print("\n--- Interactive Controls ---")
    print("Mode Selection:")
    print("  '1' : 360-degree Panorama")
    print("  '2' : Double Panorama (Front/Back stacked)")
    print("  '3' : Perspective (Virtual PTZ camera)")
    print("  '4' : Quad View (N, E, S, W perspective views)")
    print("\nVirtual PTZ Controls (in Perspective/Quad mode):")
    print("  Mouse Drag : Pan (Yaw) and Tilt (Pitch)")
    print("  Arrow Keys : Pan (Left/Right) and Tilt (Up/Down)")
    print("  '+' / '-'  : Zoom In / Zoom Out (Field of View)")
    print("\nCalibration Controls (align circular image center/radius):")
    print("  'w' / 's'  : Adjust Center Y (vertical offset)")
    print("  'a' / 'd'  : Adjust Center X (horizontal offset)")
    print("  'r' / 'f'  : Adjust Lens Radius (scale)")
    print("  '[' / ']'  : Adjust Lens Input FOV")
    print("\nNavigation:")
    print("  'space'    : Pause / Play video")
    print("  'n'        : Skip to next video (if processing folder)")
    print("  'q' or ESC : Quit")
    print("----------------------------\n")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video file reached.")
                # Loop video if not saving to output
                if not args.output:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
        else:
            # When paused, keep displaying the same frame
            pass

        # Update maps if settings changed
        if map_needs_update or map_x is None:
            if mode == 'panorama':
                map_x, map_y = make_panorama_map(W_in, H_in, width_out, height_out, Cx, Cy, R_max)
            elif mode == 'double_panorama':
                map_x, map_y = make_double_panorama_map(W_in, H_in, width_out, height_out, Cx, Cy, R_max)
            elif mode == 'perspective':
                map_x, map_y = make_perspective_map(W_in, H_in, width_out, height_out, Cx, Cy, R_max, fov_in, yaw, pitch, roll, fov_out)
            elif mode == 'quad':
                map_x, map_y = make_quad_map(W_in, H_in, width_out, height_out, Cx, Cy, R_max, fov_in, fov_out)
            
            # Print current calibration parameters
            print(f"Calibration: Center=({Cx:.1f}, {Cy:.1f}), Radius={R_max:.1f}, Lens FOV={fov_in:.1f}° | View: Mode={mode}, PTZ: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, Zoom FOV={fov_out:.1f}°", end='\r')
            map_needs_update = False

        # Apply dewarp map
        dewarped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # Draw HUD info on dewarped screen (overlay layout text)
        hud = dewarped.copy()
        cv2.putText(hud, f"Mode: {mode.upper()} (1,2,3,4 to change)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if mode == 'perspective':
            cv2.putText(hud, f"PTZ: Yaw {yaw:.1f} | Pitch {pitch:.1f} | Zoom {fov_out:.1f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(hud, f"Calib Center: {Cx:.1f},{Cy:.1f} Radius: {R_max:.1f}", (width_out - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Blend HUD with frame slightly
        cv2.addWeighted(hud, 0.7, dewarped, 0.3, 0, dewarped)

        # Write output frame if enabled
        if out_video is not None:
            out_video.write(dewarped)

        # Display window
        cv2.imshow(window_name, dewarped)

        # Handle keyboard input
        key = cv2.waitKey(delay) & 0xFF

        # Exit/Skip options
        if key == ord('q') or key == 27:  # ESC or q
            print("\nExiting program.")
            cap.release()
            if out_video:
                out_video.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('n'):  # Next file
            print("\nSkipping to next video.")
            break
        elif key == ord(' '):  # Space to pause
            paused = not paused
            print(f"\n{'Paused' if paused else 'Resumed'}")

        # Mode switching
        elif key == ord('1'):
            mode = 'panorama'
            map_needs_update = True
        elif key == ord('2'):
            mode = 'double_panorama'
            map_needs_update = True
        elif key == ord('3'):
            mode = 'perspective'
            map_needs_update = True
        elif key == ord('4'):
            mode = 'quad'
            map_needs_update = True

        # Virtual PTZ movement via keyboard
        elif key == 81 or key == 2:  # Left arrow
            yaw = (yaw - 5.0) % 360.0
            map_needs_update = True
        elif key == 83 or key == 3:  # Right arrow
            yaw = (yaw + 5.0) % 360.0
            map_needs_update = True
        elif key == 82 or key == 0:  # Up arrow
            pitch = np.clip(pitch - 5.0, 0.0, 90.0)
            map_needs_update = True
        elif key == 84 or key == 1:  # Down arrow
            pitch = np.clip(pitch + 5.0, 0.0, 90.0)
            map_needs_update = True

        # Zoom keys
        elif key == ord('+') or key == ord('='):
            fov_out = np.clip(fov_out - 5.0, 30.0, 130.0)  # Decreasing FOV zooms in
            map_needs_update = True
        elif key == ord('-') or key == ord('_'):
            fov_out = np.clip(fov_out + 5.0, 30.0, 130.0)  # Increasing FOV zooms out
            map_needs_update = True

        # Lens Calibration adjustments
        elif key == ord('a'):  # Center X left
            Cx -= 1.0
            map_needs_update = True
        elif key == ord('d'):  # Center X right
            Cx += 1.0
            map_needs_update = True
        elif key == ord('w'):  # Center Y up
            Cy -= 1.0
            map_needs_update = True
        elif key == ord('s'):  # Center Y down
            Cy += 1.0
            map_needs_update = True
        elif key == ord('r'):  # Radius larger
            R_max += 1.0
            map_needs_update = True
        elif key == ord('f'):  # Radius smaller
            R_max = max(1.0, R_max - 1.0)
            map_needs_update = True
        elif key == ord('['):  # FOV smaller
            fov_in = max(10.0, fov_in - 5.0)
            map_needs_update = True
        elif key == ord(']'):  # FOV larger
            fov_in = min(360.0, fov_in + 5.0)
            map_needs_update = True

    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Ceiling Mounted Circular Fisheye Dewarping Tool (180°–220°)")
    parser.add_argument("input", nargs="?", default=".", help="Path to input video file, folder containing videos, or camera index (default: current directory)")
    parser.add_argument("-o", "--output", default=None, help="Output file path (for single video) or folder path (for batch)")
    parser.add_argument("-m", "--mode", default="perspective", choices=["panorama", "double_panorama", "perspective", "quad"], help="Default dewarp mode (default: perspective)")
    parser.add_argument("--fov-in", type=float, default=180.0, help="Lens input FOV in degrees (default: 180)")
    parser.add_argument("--width", type=int, default=1280, help="Dewarped output width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Dewarped output height (default: 720)")
    
    # Calibration override parameters
    parser.add_argument("--cx", type=float, default=-1.0, help="Fisheye circle center X. Auto-computed if not set.")
    parser.add_argument("--cy", type=float, default=-1.0, help="Fisheye circle center Y. Auto-computed if not set.")
    parser.add_argument("--radius", type=float, default=-1.0, help="Fisheye circle radius. Auto-computed if not set.")

    # Batch option
    parser.add_argument("--batch", action="store_true", help="Process files headlessly without showing the interactive window.")

    args = parser.parse_args()

    # Determine input type
    inputs = []
    if args.input.isdigit():
        # Camera index
        inputs = [int(args.input)]
    else:
        # Check if it's a directory
        if os.path.isdir(args.input):
            # Scan for MKV, MP4, AVI, WebM files
            patterns = ["*.mkv", "*.mp4", "*.avi", "*.webm"]
            for pat in patterns:
                # Case-insensitive globbing support
                inputs.extend(glob.glob(os.path.join(args.input, pat)))
                inputs.extend(glob.glob(os.path.join(args.input, pat.upper())))
            inputs = sorted(list(set(inputs)))
            if not inputs:
                print(f"No video files found in folder: {args.input}")
                sys.exit(1)
            print(f"Found {len(inputs)} video files in folder: {args.input}")
        elif os.path.exists(args.input):
            inputs = [args.input]
        else:
            print(f"Error: Input path {args.input} does not exist.")
            sys.exit(1)

    # Process all inputs
    if args.batch:
        if not args.output:
            print("Error: --output directory must be specified for headless batch processing.")
            sys.exit(1)
        for idx, video in enumerate(inputs):
            print(f"\n[{idx+1}/{len(inputs)}] Batch processing file: {video}")
            process_video(video, args, is_headless=True)
    else:
        # Interactive mode
        for idx, video in enumerate(inputs):
            print(f"\n[{idx+1}/{len(inputs)}] Opening interactive player for: {video}")
            process_video(video, args, is_headless=False)
            print("Finished video.")

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
