# deFish

deFish is a collection of high-performance Python tools for real-time defisheye and dewarping conversion on video feeds, folders of videos, or webcam streams. It features two primary tools:

1. **Standard Defisheye (`live_defisheye.py`)**: For standard action cameras (like GoPros) and horizontal perspective correction.
2. **Ceiling-Mounted Surveillance Dewarp (`live_surveillance_dewarp.py`)**: For full-frame circular fisheye surveillance cameras (180°–220° FOV) with interactive virtual PTZ, panorama, double panorama, and quad-view projections.

---

## Features

- **Blazing Fast Conversion**: Coordinates are cached on initialization or parameters change, allowing frames to be processed in real-time (>60 FPS on CPU) using `cv2.remap`.
- **Ceiling Surveillance Dewarping**: Specially designed geometry transformations for 360-degree ceiling cameras.
- **Multiple Projection Modes**:
  - **360° Panorama**: Converts the circular view into a single wide cylindrical strip.
  - **Double Panorama**: Splits the room into stacked Front and Back 180° views.
  - **Perspective (Virtual PTZ)**: Select and interactively rotate/zoom a perspective camera frame within the sphere.
  - **Quad View**: Renders 4 virtual security cameras in a 2x2 grid looking in 4 compass directions.
- **Interactive Calibration & PTZ**: Pan/tilt with mouse drag, zoom with key controls, and align the circular lens center and radius in real-time.

---

## Installation

Install dependencies:
```bash
pip install opencv-python numpy defisheye
```

---

## Usage

### 1. Interactive Surveillance Dewarping
Run on a single file, a directory of files, or a camera index.

**Run on a specific video file:**
```bash
python live_surveillance_dewarp.py path/to/video.mkv
```

**Run on all video files in the current folder:**
```bash
python live_surveillance_dewarp.py .
```

**Run on live webcam index 0:**
```bash
python live_surveillance_dewarp.py 0
```

#### Interactive Hotkeys
With the display window focused:
- **`1`**: Switch to **Panorama** mode
- **`2`**: Switch to **Double Panorama** mode
- **`3`**: Switch to **Perspective** (Virtual PTZ) mode
- **`4`**: Switch to **Quad View** mode
- **Mouse Left Click & Drag**: Pan (Yaw) and Tilt (Pitch) in Perspective mode
- **Arrow Keys**: Pan / Tilt
- **`+` / `-`**: Zoom In / Zoom Out
- **`w` / `s`**: Adjust center vertical offset ($C_y$)
- **`a` / `d`**: Adjust center horizontal offset ($C_x$)
- **`r` / `f`**: Adjust circular lens radius size ($R_{max}$)
- **`[` / `]`**: Adjust lens input field of view (FOV)
- **`SPACE`**: Play / Pause video
- **`n`**: Skip to the next video file in the folder
- **`q` or `ESC`**: Save output video and quit

---

### 2. Headless Batch Processing
You can process all videos in a folder headlessly and export the results to an output folder.

```bash
python live_surveillance_dewarp.py ./input_folder --batch -o ./output_folder --mode double_panorama
```

Options:
- `-o`, `--output`: Target path or output directory to save MP4 files.
- `-m`, `--mode`: Default mode (`panorama`, `double_panorama`, `perspective`, `quad`).
- `--fov-in`: Input lens FOV in degrees (default: `180.0`).
- `--width` / `--height`: Output video resolution (default: `1280` x `720`).
- `--cx` / `--cy` / `--radius`: Manual calibration overrides for the fisheye center and radius.