# deFish

deFish is a Python tool that performs fast, real-time defisheye conversion on video feeds or files. It uses the `defisheye` library under the hood but wraps it with a caching mechanism to pre-calculate mapping coordinates. This ensures that processing subsequent frames is extremely fast, enabling real-time video correction without cropping or data loss.

## Features

- **Blazing Fast Conversion**: Caches mapping coordinates (xs, ys) on the first frame to ensure subsequent frames are processed in real-time.
- **Full Resolution Preservation**: Bypasses default cropping behaviors to map across the whole frame width and height, preserving the original aspect and data.
- **Flexible Input**: Can process both live webcam feeds and pre-recorded video files.
- **Customizable Parameters**: Easy adjustment of field of view (FOV), projection format (`linear`, `equalarea`, `orthographic`, `stereographic`), and padding to achieve the exact desired visual output.

## Prerequisites

Ensure you have Python 3.x installed. You will need the following Python libraries:
- `opencv-python`
- `numpy`
- `defisheye`

## Installation

1. Clone or download this repository.
2. Install the required dependencies:

```bash
pip install opencv-python numpy defisheye
```

## Usage

1. Open `live_defisheye.py` in your preferred text editor.
2. Update the input video path or set it to use your webcam.
   - For webcam: Uncomment `cap = cv2.VideoCapture(0)` and comment out the video file path.
   - For video file: Update the path in `cap = cv2.VideoCapture("path/to/your/input.mp4")`.
3. (Optional) Update the output video path in the `main()` function:
   ```python
   out_path = r"path/to/save/output_defisheye.mp4"
   ```
4. Run the script:

```bash
python live_defisheye.py
```

Press **`q`** while the video windows are focused to stop processing and save the output.

## Configuration

You can tweak the defisheye mapping in the `main()` function of `live_defisheye.py`:

- `dtype`: Projection type (`linear`, `equalarea`, `orthographic`, `stereographic`).
- `format`: Output format, e.g., `fullframe` or `circular`.
- `fov`: Field of view of the camera (e.g., 140 for many action cameras).
- `pfov`: The target field of view for the perspective projection.
- `pad`: Padding to allow capturing pixels outside standard radiuses.

## Troubleshooting
- If the video doesn't open, double-check your file paths within `live_defisheye.py`.
- For performance issues, ensure you are not running other resource-heavy applications, though the cached coordinate mapping is designed precisely to mitigate processing overhead.