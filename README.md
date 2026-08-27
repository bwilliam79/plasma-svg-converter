# Plasma SVG Converter

A web tool that converts photographs and images into CNC plasma-cut ready SVG files. Upload an image, configure cut parameters, and download a clean SVG with properly simplified paths, interior cutouts, and automatic bridge insertion.

---

## Features

### Subject Detection
- Automatically removes the background using multiple fallback methods:
  1. PNG alpha channel (if present)
  2. AI background removal via `rembg`
  3. OpenCV GrabCut segmentation
  4. Otsu thresholding (last resort)

### Interior Hole Recovery
- Detects interior cutouts that background removal may accidentally fill
- Uses saturation analysis and seeded region growing to distinguish holes from the subject
- Edge-proximity filtering removes noise without losing legitimate holes

### Path Simplification
- Two-stage smoothing: moving-average pass followed by Douglas-Peucker reduction
- Adjustable simplification factor (1–10) to trade detail for path complexity

### Bridge Insertion
- Automatically adds bridge gaps to interior cutouts to prevent parts from falling during cutting
- Configurable bridge width (mm) and bridge count per hole
- Evenly distributed along the contour using arc-length parameterization

### Output
- SVG scaled to a specified output width (default 200mm), aspect ratio preserved
- Preview shows colored cut paths and cyan bridge markers
- Downloaded SVG is clean (bridge markers stripped) and ready for CAM software

---

## Setup

### Prerequisites
- Docker and Docker Compose

### Start

```bash
docker compose up -d
```

Available at **http://localhost:5001**.

> On first run, `rembg` will download its background-removal model (~170MB). This is cached in the `rembg-models` Docker volume and only happens once.

---

## Running Without Docker

```bash
pip install -r requirements.txt
python app.py
# Runs at http://localhost:5000
```

Requires Python 3.11+. Install system dependencies for OpenCV if not already present (`libgl1`, `libglib2.0-0` on Debian/Ubuntu).

---

## Parameters

All parameters are configurable in the UI:

| Parameter | Default | Description |
|---|---|---|
| Output width | 200mm | Final SVG width (height scales proportionally) |
| Simplification | 2.0 | Path simplification aggressiveness (1–10) |
| Keep holes | On | Whether to preserve interior cutouts |
| Bridge width | 3.0mm | Width of each bridge gap |
| Bridges per hole | 3 | Number of bridges inserted per interior cutout |

---

## Tech Stack

- **Backend**: Python 3.11, Flask, OpenCV, rembg (AI background removal), NumPy, Pillow
- **Frontend**: Vanilla JavaScript, HTML/CSS
- **Deployment**: Docker Compose

## PWA icons

App icon is the locked plasma kerf-bear mark (`a-kerf-bear`). Pack in `static/` (Flask default static):

- `favicon.ico` / `favicon-16.png` / `favicon-32.png`
- `apple-touch-icon.png` (180)
- `icon-192.png` / `icon-512.png` / `icon-maskable-512.png`
- `manifest.json`

Linked from `templates/index.html` as `/static/…?v=`.

