import cv2
import numpy as np
from defisheye import Defisheye

class FastDefisheye(Defisheye):
    """
    A wrapper around Defisheye that caches the mapping coordinates (xs, ys)
    so that real-time video processing is extremely fast.
    """
    def __init__(self, infile, **kwargs):
        super().__init__(infile, **kwargs)
        self._xs = None
        self._ys = None
        
        # No need to pre-calculate crop dimensions if we don't want to crop the output!
        # The parent init sets these up, but we'll override it to map across the whole frame width/height.
        self.width = infile.shape[1]
        self.height = infile.shape[0]

    def build_map(self):
        # We calculate the required output grid without forcing it into a square `dim`

        # Base focal calculations
        if self._format == "circular":
            dim = min(self.width, self.height)
        elif self._format == "fullframe":
            # For wide video, basing dim on width ensures horizontal FoV is properly captured
            dim = float(self.width)

        if self._radius is not None:
            dim = 2 * self._radius

        ofoc = dim / (2 * np.tan(self._pfov * np.pi / 360))
        ofocinv = 1.0 / ofoc

        # Instead of self._width which is cropped, use self.width
        i = np.arange(self.width)
        j = np.arange(self.height)
        i, j = np.meshgrid(i, j)

        self._xs, self._ys = self._fast_map(i, j, ofocinv, dim)

    def _fast_map(self, i, j, ofocinv, dim):
        # We need the true centers
        xcenter = self.width // 2
        ycenter = self.height // 2

        xd = i - xcenter
        yd = j - ycenter

        rd = np.hypot(xd, yd)
        phiang = np.arctan(ofocinv * rd)

        if self._dtype == "linear":
            ifoc = dim * 180 / (self._fov * np.pi)
            rr = ifoc * phiang
        elif self._dtype == "equalarea":
            ifoc = dim / (2.0 * np.sin(self._fov * np.pi / 720))
            rr = ifoc * np.sin(phiang / 2)
        elif self._dtype == "orthographic":
            ifoc = dim / (2.0 * np.sin(self._fov * np.pi / 360))
            rr = ifoc * np.sin(phiang)
        elif self._dtype == "stereographic":
            ifoc = dim / (2.0 * np.tan(self._fov * np.pi / 720))
            rr = ifoc * np.tan(phiang / 2)

        rdmask = rd != 0
        xs = xd.astype(np.float32).copy()
        ys = yd.astype(np.float32).copy()

        # The parent logic uses self._xcenter which is cropped. We use true xcenter here.
        # But importantly, the xd being multiplied might map out of bounds in the original _image
        # Because we're not cropping _image, the _image is full res.
        xs[rdmask] = (rr[rdmask] / rd[rdmask]) * xd[rdmask] + xcenter
        ys[rdmask] = (rr[rdmask] / rd[rdmask]) * yd[rdmask] + ycenter

        xs[~rdmask] = 0
        ys[~rdmask] = 0

        return xs, ys

    def fast_convert(self, frame):
        # Do not crop the frame, map the full resolution
        return cv2.remap(frame, self._xs, self._ys, cv2.INTER_LINEAR)


def main():
    # Defisheye parameters
    dtype = 'linear'
    format = 'fullframe'
    fov = 140   # Assuming an action camera like GoPro
    pfov = 140  # Matching pfov to fov ensures 100% of the horizontal edge pixels are captured
    pad = 0    # Padding allows capturing pixels outside the standard radius

    # Initialize video capture
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"D:\RSSB\DCC\fisheye\fisheye_video.mp4")

    if not cap.isOpened():
        print("Error: Could not open camera or video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0
    delay = int(1000 / fps)
    
    print(f"Video FPS: {fps}, delay per frame: {delay} ms")
    print("Press 'q' to quit.")
    
    fast_defisheye = None
    out_video = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        if fast_defisheye is None:
            # Initialize on the first frame
            fast_defisheye = FastDefisheye(frame, dtype=dtype, format=format, fov=fov, pfov=pfov, pad=pad)
            fast_defisheye.build_map()

        # Perform fast conversion using the cached map
        out_frame = fast_defisheye.fast_convert(frame)
        
        # Display input and output in separate windows
        try:
            # Initialize video writer on the first frame
            if out_video is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width = out_frame.shape[:2]
                out_path = r"D:\RSSB\DCC\fisheye\output_defisheye.mp4"
                print(f"Saving output video to: {out_path} with size {(width, height)}")
                out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                
            out_video.write(out_frame)
            
            # Show Original and Defisheyed windows separately
            cv2.imshow('Original Input', frame)
            cv2.imshow('Corrected Output', out_frame)
            
        except Exception as e:
            print(f"Display error: {e}")
            cv2.imshow('Original', frame)
            cv2.imshow('Defisheyed', out_frame)
            
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    if out_video is not None:
        out_video.release()
        print("Output video saved.")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
