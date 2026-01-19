# Blob Tracker - Standalone Application

High-performance blob tracking with real-time visualization. Optimized for 60-120 FPS with advanced features like motion smoothing, connection networks, and tactical HUD overlays.


## Features

- **Real-time blob detection** - Track multiple objects simultaneously
- **Motion smoothing** - Natural interpolation with cubic easing
- **Connection networks** - Visual links between nearby blobs
- **Tactical HUD** - Corner brackets, metrics overlay, tracking data
- **Trail rendering** - Smooth or dotted motion trails
- **Grid overlay** - Technical tracking aesthetic
- **High performance** - 60-120 FPS with frame skipping and adaptive resolution

## Installation

### Requirements
- Python 3.7+
- OpenCV
- NumPy

### Install Dependencies

```bash
pip install opencv-python numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Webcam)

```bash
python blob_tracker_app.py
```

### From Video File

```bash
python blob_tracker_app.py --input video.mp4
```

### Save Output

```bash
python blob_tracker_app.py --output tracked_output.mp4
```

### Adjust Detection Parameters

```bash
python blob_tracker_app.py --threshold 150 --min-area 20 --max-area 500 --max-blobs 50
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `t` | Toggle trails |
| `c` | Toggle connection lines |
| `b` | Toggle corner brackets |
| `m` | Toggle metrics overlay |
| `g` | Toggle grid overlay |
| `d` | Toggle dotted line style |

## Configuration

Edit the `config` dictionary in `blob_tracker_app.py` to customize:

```python
config = {
    'threshold_value': 127,          # Blob detection threshold
    'min_area': 10,                  # Minimum blob size
    'max_area': 1000,                # Maximum blob size
    'max_blobs': 100,                # Max blobs to track
    'resolution_scale': 0.75,        # Detection resolution (0.5-1.0)
    'enable_skip': True,             # Frame skipping for speed
    'frame_skip_interval': 2,        # Frames between detections
    'motion_smoothing': 0.5,         # Motion interpolation (0.0-1.0)
    'size_smoothing': 0.5,           # Size stability (0.0-1.0)
    'outline_color': (0, 255, 255),  # BGR color for blobs
    'trail_color': (0, 255, 255),    # BGR color for trails
    'blob_thickness': 2,             # Line thickness
    'draw_connections': True,        # Show connection network
    'draw_trails': True,             # Show motion trails
    'line_smoothness': 8,            # Trail smoothness (2-16)
    'max_line_length': 1.0,          # Trail length (0.0-1.0)
    'show_ids': True,                # Show blob IDs
    'show_leaders': False,           # Lines from blob to label
    'show_metrics': True,            # Data overlay panel
    'show_grid': False,              # Grid overlay
    'grid_spacing': 50.0,            # Grid cell size
    'use_brackets': True,            # Corner brackets vs full boxes
    'bracket_length': 0.3,           # Bracket size (0.1-0.5)
    'use_dotted': False,             # Dotted line style
}
```

## Performance Tuning

### For Maximum Speed (120 FPS+)
```python
'resolution_scale': 0.5,
'frame_skip_interval': 3,
'enable_skip': True,
'line_smoothness': 4,
'draw_trails': False,
```

### For Best Quality (60 FPS)
```python
'resolution_scale': 1.0,
'frame_skip_interval': 2,
'enable_skip': True,
'line_smoothness': 12,
'draw_trails': True,
```

### For Surveillance Look
```python
'outline_color': (0, 255, 255),  # Cyan
'use_brackets': True,
'show_metrics': True,
'show_grid': True,
'draw_connections': True,
'use_dotted': True,
```

## Command Line Arguments

```
--input         Input source (camera index or video file, default: 0)
--output        Output video file (optional)
--threshold     Threshold value for blob detection (default: 127)
--min-area      Minimum blob area in pixels (default: 10)
--max-area      Maximum blob area in pixels (default: 1000)
--max-blobs     Maximum number of blobs to track (default: 100)
```

## Examples

### Track from webcam with cyan surveillance aesthetic
```bash
python blob_tracker_app.py
```

### Process video file and save output
```bash
python blob_tracker_app.py --input input.mp4 --output tracked.mp4
```

### Detect smaller objects with higher sensitivity
```bash
python blob_tracker_app.py --threshold 100 --min-area 5 --max-area 200
```

### Track many fast-moving objects
```bash
python blob_tracker_app.py --max-blobs 200 --threshold 150
```

## Troubleshooting

### Low FPS
- Lower `resolution_scale` (try 0.5)
- Increase `frame_skip_interval` (try 3-4)
- Reduce `max_blobs`
- Disable `draw_trails`
- Lower `line_smoothness`

### Blobs not detected
- Adjust `threshold` value (try both higher and lower)
- Check `min_area` and `max_area` match your object sizes
- Try toggling `invert_threshold` in config

### Jittery motion
- Increase `motion_smoothing` (try 0.7)
- Increase `size_smoothing` (try 0.7)
- Lower `frame_skip_interval`

### Connection lines too dense/sparse
- Connections auto-adjust to 3x blob size
- Change detection parameters to get differently sized blobs

## Project Structure

```
blob-tracker/
├── blob_tracker_app.py    # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── examples/             # Example videos/images
```

## Technical Details

### Optimizations
- Cached blob detector (avoids recreation each frame)
- Vectorized NumPy operations
- Frame skip with motion interpolation
- Adaptive resolution detection
- Buffer reuse for output images

### Smoothing Techniques
- Cubic easing for natural acceleration
- Exponential velocity smoothing
- Position smoothing for stability
- Size smoothing to prevent pulsing

### Performance
- 60-80 FPS: Full resolution, all features enabled
- 90-110 FPS: 0.75 resolution scale, frame skip 2
- 120+ FPS: 0.5 resolution scale, frame skip 3-4

## License

N/A

## Credits

Created by nicholaspjm

## Contributing

Pull requests welcome! Areas for improvement:
- GPU acceleration (CUDA)
- Multi-threaded processing
- Custom blob detectors
- Additional visual effects
- Web interface

## Contact

For questions or feature requests, open an issue on GitHub.
