# YOLO Rect Annotator

A simple **Python + OpenCV** tool for annotating **rectangular bounding boxes (BBOX)** in **video frames or images** and exporting them in **YOLO format**.  
After drawing each box, you type the **numeric class ID** in the terminal.

---

## ✨ Features
- Annotate a **video** or a **set of images** (folder, glob pattern, or a single file) with the same tool.
- Interactive annotation of multiple BBOX per frame/image.
- Manual input of **class ID** after each drawn BBOX.
- Output in **YOLOv5/YOLOv8 format**:

All coordinates are **normalized** (0–1).
- Keyboard shortcuts:
- Save (`s`)
- Skip frame/image (`n`)
- Undo last BBOX (`u`)
- Remove all BBOX in current frame/image (`r`)
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

### Video

```bash
python annotate_video_yolo.py \
  --video /path/to/video.mp4 \
  --outdir dataset \
  --img-prefix myke \
  --skip 0 \
  --start 0
```

### Images (folder, glob pattern, or single file)

```bash
# a whole folder of images (sorted by filename)
python annotate_video_yolo.py --images /path/to/images_folder --outdir dataset --img-prefix myke

# skip 1 image between annotations, starting at the 6th image (index 5)
python annotate_video_yolo.py --images /path/to/images_folder --outdir dataset --skip 1 --start 5

# a glob pattern
python annotate_video_yolo.py --images "/path/to/photos/*.jpg" --outdir dataset

# a single image
python annotate_video_yolo.py --images /path/to/photo.png --outdir dataset
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp` (case-insensitive; other files in the folder are ignored).

Each image produces `dataset/images/<img-prefix>-<original_filename>.jpg` and `dataset/labels/<img-prefix>-<original_filename>.txt`, preserving the original filename for traceability.

## Arguments

--video: path to input video. Mutually exclusive with `--images`; exactly one of the two is required.

--images: path to a folder of images, a glob pattern (e.g. `"photos/*.jpg"`), or a single image file. Mutually exclusive with `--video`.

--outdir (default: dataset): output root directory. It will contain:

--img-prefix (default: frame): prefix for .jpg/.txt files.

--skip (default: 0): skip N frames/images between annotations (0 = annotate every one).

--start (default: 0): starting frame/image index (0-based).

## ⌨️ Keyboard Controls

q → quit

s → save image + labels for the current frame/image

n → go to next frame/image (discard unsaved BBOX)

u → undo last BBOX

r → remove all BBOX in the current frame/image
