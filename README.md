# 🗑️ Smart Garbage & Road Hazard Detection AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-red.svg)](https://opencv.org/)
[![Rich](https://img.shields.io/badge/Terminal%20UI-Rich-purple.svg)](https://github.com/Textualize/rich)

An intelligent computer-vision application for detecting **Garbage**, **Potholes**, **Road Cracks**, and **Manholes** with real-time HUD visualization, calibrated sensitivity, and false-positive suppression.

---

## ✨ Features

- **🎨 Modern Visual Annotations**:
  - High-tech corner targeting brackets on bounding boxes.
  - Translucent alpha-tinted object highlights.
  - Rounded confidence pill badges.
  - Floating top dashboard HUD with live hazard counters and inference latency.
- **🛡️ Smart Hazard Guard**:
  - **Road Marking Rejection**: Intelligently identifies and ignores painted road lane lines, stripes, and crosswalks so they are never falsely detected as garbage.
  - **Pothole Cavity Auditing**: Differentiates dark asphalt depressions, puddles, and road craters from surface garbage.
  - **Full-Frame Hallucination Filter**: Suppresses oversized background artifacts.
- **💻 Rich Terminal UI**:
  - Color-coded detection tables with visual confidence progress meters.
  - Real-time diagnostic metrics (preprocess, inference, and postprocess ms).
- **📂 Interactive Source Picker**:
  - Choose between a native GUI file browser dialog, live webcam feed (`0`), or manual path input.

---

## 🚀 Quick Start

### 1. Installation

Ensure you have Python installed, then install the dependencies:

```bash
pip install ultralytics opencv-python rich numpy pillow
```

### 2. Run Inference

#### Interactive Mode (GUI File Picker / Webcam)
```bash
python predict.py
```

#### Run on a Specific Image or Video
```bash
python predict.py --source "path/to/media.jpg" --save
```

#### Run on Live Webcam
```bash
python predict.py --source 0
```

---

## ⚙️ CLI Options

| Option | Default | Description |
| :--- | :---: | :--- |
| `--model` | `best.pt` | Path to trained model weights |
| `--source` | `None` | Image/video file path, webcam (`0`), or RTSP stream |
| `--conf` | `0.25` | Confidence threshold ($0.0 - 1.0$) |
| `--iou` | `0.45` | NMS IoU threshold for overlapping detections |
| `--augment` | `False` | Enable Test-Time Augmentation (TTA) for difficult scenes |
| `--save` | `False` | Save annotated output images/videos to output folder |
| `--out-dir` | `runs/predict` | Output directory for saved results |
| `--no-filter` | `False` | Disable automatic road marking and false-alarm suppression |
| `--no-view` | `False` | Run in headless mode without opening GUI window |
| `--no-hud` | `False` | Hide top dashboard banner on images |

---

## 🏷️ Detected Classes

- 🗑️ **Garbage** (Litter, plastic bags, drink cups, bottles, food wrappers)
- 🕳️ **Pothole** (Pavement cavities, road depressions, craters)
- ⚡ **Crack** (Road surface fractures and structural fissures)
- 🔘 **Manhole** (Drain covers and utility access rings)

---

## 📜 License
MIT License. Created by [Aditya Kollapudi](https://github.com/adityakollapudi).
