# YOLO Rect Annotator

A simple **Python + OpenCV** tool for annotating **rectangular bounding boxes (BBOX)** in video frames and exporting them in **YOLO format**.  
After drawing each box, you type the **numeric class ID** in the terminal.

---

## ✨ Features
- Interactive annotation of multiple BBOX per frame.
- Manual input of **class ID** after each drawn BBOX.
- Output in **YOLOv5/YOLOv8 format**:

All coordinates are **normalized** (0–1).
- Keyboard shortcuts:
- Save (`s`)
- Skip frame (`n`)
- Undo last BBOX (`u`)
- Remove all BBOX in frame (`r`)
- Quit (`q`)

---

## 📦 Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy
```

## ▶️ Usage

```bash
python annotate_video_yolo.py \
  --video /path/to/video.mp4 \
  --outdir dataset \
  --img-prefix myke \
  --skip 0 \
  --start 0
```

## Arguments

--video (required): path to input video.

--outdir (default: dataset): output root directory. It will contain:

--img-prefix (default: frame): prefix for .jpg/.txt files.

--skip (default: 0): skip N frames between annotations (0 = annotate every frame).

--start (default: 0): starting frame index (0-based).

## ⌨️ Keyboard Controls

q → quit

s → save image + labels for the current frame

n → go to next frame (discard unsaved BBOX)

u → undo last BBOX

r → remove all BBOX in the current frame
