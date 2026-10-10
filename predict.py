"""
=============================================================================
  AI GARBAGE & ROAD HAZARD DETECTION SYSTEM
  Powered by YOLOv8 & OpenCV High-Aesthetic Visualization Engine
  With Smart Pothole Auditing & False-Positive Guard
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# -----------------------------------------------------------------------------
# Color Palette & Styling Tokens (BGR Format for OpenCV)
# -----------------------------------------------------------------------------
CLASS_PALETTE = {
    "Garbage": {
        "color": (52, 73, 245),       # Vibrant Coral / Red
        "badge_bg": (35, 45, 170),
        "emoji": "🗑️",
        "priority": "HIGH",
        "style": "bold red"
    },
    "Pothole": {
        "color": (26, 140, 255),      # Electric Amber / Orange
        "badge_bg": (15, 95, 180),
        "emoji": "🕳️",
        "priority": "CRITICAL",
        "style": "bold yellow"
    },
    "Crack": {
        "color": (0, 215, 255),       # Neon Gold / Yellow
        "badge_bg": (0, 145, 175),
        "emoji": "⚡",
        "priority": "MEDIUM",
        "style": "bold cyan"
    },
    "Manhole": {
        "color": (216, 50, 180),      # Bright Violet / Magenta
        "badge_bg": (150, 30, 125),
        "emoji": "🔘",
        "priority": "LOW",
        "style": "bold magenta"
    }
}

DEFAULT_STYLE = {
    "color": (50, 220, 100),          # Emerald Green
    "badge_bg": (30, 150, 70),
    "emoji": "📦",
    "priority": "INFO",
    "style": "bold green"
}


def get_class_meta(class_name: str) -> dict:
    """Returns aesthetic metadata and color configuration for a class."""
    for key, val in CLASS_PALETTE.items():
        if key.lower() == class_name.lower():
            return val
    return DEFAULT_STYLE


def print_banner():
    """Prints a styled, modern terminal banner."""
    banner_text = Text()
    banner_text.append("╔══════════════════════════════════════════════════════════════════════╗\n", style="bold cyan")
    banner_text.append("║           ✨ SMART GARBAGE & ROAD HAZARD DETECTOR AI ✨              ║\n", style="bold white")
    banner_text.append("║           Real-Time Object Detection & Visual Intelligence           ║\n", style="bold bright_blue")
    banner_text.append("╚══════════════════════════════════════════════════════════════════════╝", style="bold cyan")
    console.print(Panel(banner_text, border_style="cyan", padding=(0, 2)))


# -----------------------------------------------------------------------------
# Targeted False-Positive & Road-Marking Rejection Filter
# -----------------------------------------------------------------------------
def check_false_positive(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    class_name: str,
    conf: float,
    filter_markings: bool = True,
    max_area_ratio: float = 0.70
) -> Tuple[bool, str]:
    """
    Carefully suppresses road paint markings and full-frame hallucinations
    WITHOUT discarding real garbage (cups, bags, bottles, bowls).
    """
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    ih, iw = image.shape[:2]
    box_area = bw * bh
    total_area = iw * ih

    # 1. Reject full-frame background hallucinations (> 70% of entire scene)
    if box_area / total_area > max_area_ratio:
        return True, f"Full-frame background coverage ({int(box_area/total_area*100)}% of image)"

    # 2. Reject micro noise speckles (< 10 px)
    if bw < 10 or bh < 10:
        return True, "Micro noise artifact"

    # 3. Road lane marking & paint stripe rejection
    if filter_markings and class_name.lower() == "garbage":
        aspect_ratio = max(bh / bw, bw / bh)
        
        # Elongated straight stripe (Lane divider, zebra stripe, painted curb mark)
        if aspect_ratio >= 2.8:
            crop = image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                std_brightness = float(np.std(gray))
                
                # Painted road lines are high brightness (mean > 150) with uniform paint surface
                if mean_brightness > 150 and std_brightness < 45:
                    return True, f"Road paint line stripe (Aspect ratio: {aspect_ratio:.1f}, Brightness: {mean_brightness:.0f})"

    return False, ""


# -----------------------------------------------------------------------------
# Smart Pothole Audit & Reclassification Engine
# -----------------------------------------------------------------------------
def audit_and_reclassify_detection(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    class_name: str,
    conf: float
) -> Tuple[str, float, Tuple[int, int, int, int], Optional[str]]:
    """
    Corrects model misclassifications:
    Road craters, asphalt depressions, and water puddles falsely labeled as
    'Garbage' due to dataset training bias are restored to their true class: 'Pothole'.
    """
    x1, y1, x2, y2 = box
    ih, iw = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    if class_name.lower() == "garbage":
        crop = image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
        if crop.size > 0:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))

            # Check surrounding rim for dark asphalt crater depression
            pad_x, pad_y = int(bw * 0.35), int(bh * 0.35)
            ry1, ry2 = max(0, y1 - pad_y), min(ih, y2 + pad_y)
            rx1, rx2 = max(0, x1 - pad_x), min(iw, x2 + pad_x)
            surr = image[ry1:ry2, rx1:rx2]
            surr_gray = cv2.cvtColor(surr, cv2.COLOR_BGR2GRAY)
            dark_rim_ratio = float(np.sum(surr_gray < 100)) / max(1, surr_gray.size)

            # Overall scene saturation check (road scenes have desaturated asphalt)
            full_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            full_sat = float(np.mean(full_hsv[:, :, 1]))

            # Pothole signatures: Desaturated road scene, low internal chroma (water/mud/asphalt), dark crater rim
            if full_sat < 42 and sat_mean < 35 and dark_rim_ratio > 0.18:
                # Expand box slightly to encapsulate the entire road crater rim
                nx1 = max(0, x1 - int(bw * 0.20))
                ny1 = max(0, y1 - int(bh * 0.28))
                nx2 = min(iw, x2 + int(bw * 0.20))
                ny2 = min(ih, y2 + int(bh * 0.20))
                calibrated_conf = max(conf, 0.82)
                return "Pothole", calibrated_conf, (nx1, ny1, nx2, ny2), "Reclassified from Garbage to Pothole (Road Crater Signature)"

    return class_name, conf, (x1, y1, x2, y2), None


def draw_hud(
    image: np.ndarray,
    detections_summary: Dict[str, int],
    inference_ms: float,
    fps: Optional[float] = None
) -> np.ndarray:
    """
    Draws a sleek, semi-transparent HUD dashboard banner at the top of the image.
    """
    h, w = image.shape[:2]
    hud_h = max(42, int(h * 0.065))
    
    # Create translucent dark overlay for top bar
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 20, 24), -1)
    cv2.line(overlay, (0, hud_h), (w, hud_h), (60, 65, 75), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    # Left: Title & Logo
    logo_text = "HAZARD AI VISION"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.45, hud_h / 85.0)
    thickness = 1
    
    cv2.putText(image, logo_text, (18, int(hud_h * 0.65)), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

    # Center: Summary pill counts
    current_x = int(w * 0.32)
    if not detections_summary:
        cv2.putText(
            image,
            "No Hazards / Clean Road",
            (current_x, int(hud_h * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (140, 220, 140),
            1,
            cv2.LINE_AA
        )
    else:
        for cls_name, count in detections_summary.items():
            meta = get_class_meta(cls_name)
            color = meta["color"]
            
            # Dot indicator
            dot_y = int(hud_h * 0.52)
            cv2.circle(image, (current_x, dot_y), int(hud_h * 0.12), color, -1, cv2.LINE_AA)
            
            # Text
            badge_txt = f"{cls_name}: {count}"
            (t_w, _), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, 1)
            cv2.putText(
                image,
                badge_txt,
                (current_x + int(hud_h * 0.22), int(hud_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.9,
                (225, 230, 235),
                1,
                cv2.LINE_AA
            )
            current_x += t_w + int(hud_h * 0.7)

    # Right: Latency / FPS info
    perf_text = f"{inference_ms:.1f}ms"
    if fps is not None:
        perf_text = f"{fps:.1f} FPS | " + perf_text
    
    (p_w, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, 1)
    cv2.putText(
        image,
        perf_text,
        (w - p_w - 20, int(hud_h * 0.65)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.85,
        (130, 220, 130),
        1,
        cv2.LINE_AA
    )

    return image


def draw_modern_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    class_name: str,
    confidence: float
) -> np.ndarray:
    """
    Renders high-aesthetic bounding boxes with:
    1. Subtle translucent interior fill
    2. Clean antialiased border
    3. Modern corner targeting brackets (HUD style)
    4. Rounded label badge with crisp contrast text
    """
    x1, y1, x2, y2 = box
    w_box = x2 - x1
    h_box = y2 - y1
    meta = get_class_meta(class_name)
    color = meta["color"]

    # 1. Subtle alpha tinted box fill
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, image)

    # 2. Main border
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

    # 3. Corner brackets
    corner_len = min(22, max(8, int(min(w_box, h_box) * 0.18)))
    corner_thick = 3

    # Top-Left
    cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, corner_thick, cv2.LINE_AA)
    cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, corner_thick, cv2.LINE_AA)
    # Top-Right
    cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, corner_thick, cv2.LINE_AA)
    cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, corner_thick, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, corner_thick, cv2.LINE_AA)
    cv2.line(image, (x1, y2), (x1 - corner_len, y2), color, corner_thick, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, corner_thick, cv2.LINE_AA)
    cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, corner_thick, cv2.LINE_AA)

    # 4. Modern Badge
    label = f"{class_name}  {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

    pad_x, pad_y = 9, 6
    badge_w = text_w + pad_x * 2
    badge_h = text_h + pad_y * 2

    # Place badge above box if space permits, else inside
    if y1 - badge_h >= 0:
        bx1, by1 = x1, y1 - badge_h
    else:
        bx1, by1 = x1, y1

    bx2 = bx1 + badge_w
    by2 = by1 + badge_h

    # Badge background
    badge_overlay = image.copy()
    cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), meta["badge_bg"], -1)
    cv2.addWeighted(badge_overlay, 0.90, image, 0.10, 0, image)
    cv2.rectangle(image, (bx1, by1), (bx2, by2), color, 1, lineType=cv2.LINE_AA)

    # Badge text
    cv2.putText(
        image,
        label,
        (bx1 + pad_x, by1 + badge_h - pad_y),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return image


def annotate_image(
    image: np.ndarray,
    boxes_data: List[dict],
    inference_ms: float,
    hud: bool = True,
    fps: Optional[float] = None
) -> np.ndarray:
    """Annotates image with all detected boxes and optional HUD dashboard."""
    annotated = image.copy()
    counts = {}

    for item in boxes_data:
        class_name = item["class_name"]
        counts[class_name] = counts.get(class_name, 0) + 1
        draw_modern_box(
            annotated,
            item["box"],
            class_name,
            item["conf"]
        )

    if hud:
        draw_hud(annotated, counts, inference_ms, fps=fps)

    return annotated


def display_results_table(
    boxes_data: List[dict],
    inference_speed: dict,
    rejected_items: Optional[List[dict]] = None,
    reclassified_items: Optional[List[dict]] = None
):
    """Prints a beautiful, rich terminal table of detected items and diagnostics."""
    if reclassified_items:
        reclass_text = "\n".join([f"• [bold yellow]{r['old_class']} ➔ {r['new_class']}:[/bold yellow] {r['reason']}" for r in reclassified_items])
        console.print(
            Panel(
                f"[bold cyan]🔍 AI Inspector Corrected {len(reclassified_items)} Road Hazard Classifications:[/bold cyan]\n{reclass_text}",
                title="Smart Hazard Classifier Guard",
                border_style="yellow"
            )
        )

    if rejected_items:
        rejection_text = "\n".join([f"• [dim yellow]{r['reason']}[/dim yellow] ({r['conf']*100:.1f}%)" for r in rejected_items])
        console.print(
            Panel(
                f"[bold green]🛡️ Smart Guard Suppressed {len(rejected_items)} Road False-Alarms:[/bold green]\n{rejection_text}",
                title="Smart Road-Marking Guard",
                border_style="cyan"
            )
        )

    if not boxes_data:
        console.print(
            Panel(
                Text("🍃 No genuine road hazards or garbage detected in this frame. (Road clean & safe)", style="bold green"),
                title="Detection Summary",
                border_style="green"
            )
        )
        return

    table = Table(
        title=f"🎯 Detected Objects ({len(boxes_data)} Total)",
        header_style="bold cyan",
        border_style="bright_black",
        show_lines=True
    )

    table.add_column("#", justify="center", style="dim", width=4)
    table.add_column("Object Class", style="bold white", width=16)
    table.add_column("Confidence", justify="left", width=22)
    table.add_column("Priority", justify="center", width=12)
    table.add_column("Bounding Box [x1, y1, x2, y2]", justify="center", width=24)
    table.add_column("Box Size", justify="center", width=12)

    for i, item in enumerate(boxes_data, 1):
        meta = get_class_meta(item["class_name"])
        conf = item["conf"]
        conf_pct = conf * 100
        
        # Color-coded progress bar string
        bar_len = 10
        filled = int(conf * bar_len)
        bar_visual = "█" * filled + "░" * (bar_len - filled)
        
        if conf >= 0.70:
            conf_str = f"[bold green]{bar_visual} {conf_pct:.1f}%[/bold green]"
        elif conf >= 0.40:
            conf_str = f"[bold yellow]{bar_visual} {conf_pct:.1f}%[/bold yellow]"
        else:
            conf_str = f"[bold bright_cyan]{bar_visual} {conf_pct:.1f}%[/bold bright_cyan]"

        priority_color = {
            "CRITICAL": "bold red",
            "HIGH": "bold bright_red",
            "MEDIUM": "bold yellow",
            "LOW": "cyan",
            "INFO": "green"
        }.get(meta["priority"], "white")

        x1, y1, x2, y2 = item["box"]
        w_box, h_box = x2 - x1, y2 - y1

        table.add_row(
            str(i),
            f"{meta['emoji']} {item['class_name']}",
            conf_str,
            f"[{priority_color}]{meta['priority']}[/{priority_color}]",
            f"[{x1}, {y1}, {x2}, {y2}]",
            f"{w_box} x {h_box} px"
        )

    console.print(table)

    # Diagnostic Speed Panel
    prep = inference_speed.get("preprocess", 0.0)
    inf = inference_speed.get("inference", 0.0)
    post = inference_speed.get("postprocess", 0.0)
    total_ms = prep + inf + post
    fps = 1000.0 / total_ms if total_ms > 0 else 0

    speed_info = (
        f"[bold cyan]Preprocess:[/bold cyan] {prep:.1f}ms  │  "
        f"[bold yellow]Inference:[/bold yellow] {inf:.1f}ms  │  "
        f"[bold green]Postprocess:[/bold green] {post:.1f}ms  │  "
        f"[bold white]Total:[/bold white] {total_ms:.1f}ms ([bold bright_green]{fps:.1f} FPS[/bold bright_green])"
    )
    console.print(Panel(speed_info, title="⚡ Inference Diagnostics", border_style="blue"))


def select_source_interactive() -> str:
    """Provides an interactive, intuitive menu to select input source."""
    console.print(
        Panel(
            "[bold white]Please choose how you would like to run inference:[/bold white]\n\n"
            "  [bold cyan]1[/bold cyan] 📁 [bold]Browse File[/bold] (Open GUI file picker dialog)\n"
            "  [bold cyan]2[/bold cyan] 📷 [bold]Live Webcam[/bold] (Stream from Camera 0)\n"
            "  [bold cyan]3[/bold cyan] ✍️ [bold]Enter Path/URL[/bold] (Type file path or video stream URL)",
            title="🎯 Input Source Selector",
            border_style="cyan"
        )
    )

    choice = console.input("[bold yellow]Enter your choice (1, 2, or 3) [default: 1]: [/bold yellow]").strip()

    if choice == "2":
        return "0"
    elif choice == "3":
        path = console.input("[bold yellow]Enter image/video path or RTSP URL: [/bold yellow]").strip()
        return path
    else:
        # Default: Tkinter file dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            console.print("[dim cyan]Opening file dialog...[/dim cyan]")
            selected = filedialog.askopenfilename(
                title="Select Image or Video for Detection",
                filetypes=[
                    ("All Media Files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                    ("All Files", "*.*")
                ]
            )
            root.destroy()
            if selected:
                return selected
        except Exception as e:
            console.print(f"[dim red]GUI file picker unavailable ({e}), falling back to text prompt.[/dim red]")
        
        path = console.input("[bold yellow]Enter path to image or video: [/bold yellow]").strip()
        return path


def resolve_model_path(model_path: str) -> str:
    """Intelligently resolves model weights path checking local dirs."""
    candidates = [
        model_path,
        os.path.join(os.path.dirname(__file__), model_path),
        os.path.join(os.path.dirname(__file__), "Garbage-detection-main", model_path),
        os.path.join(os.path.dirname(__file__), "best.pt"),
        os.path.join(os.path.dirname(__file__), "Garbage-detection-main", "best.pt"),
        os.path.join(os.getcwd(), "best.pt"),
        os.path.join(os.getcwd(), "Garbage-detection-main", "best.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return model_path


def process_image(
    model: YOLO,
    image_path: str,
    conf_thresh: float,
    iou_thresh: float,
    filter_markings: bool,
    augment: bool,
    save: bool,
    output_dir: Path,
    view: bool,
    hud: bool
):
    """Runs inference on a single image with smart pothole auditing."""
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        console.print(f"[bold red]❌ Error: Unable to read image at '{image_path}'[/bold red]")
        return

    with console.status("[bold cyan]Running AI inference...", spinner="dots"):
        results = model.predict(
            source=orig_img,
            conf=conf_thresh,
            iou=iou_thresh,
            augment=augment,
            verbose=False
        )
    
    result = results[0]
    boxes_data = []
    rejected_items = []
    reclassified_items = []
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = model.names.get(cls_id, f"Class_{cls_id}")

        # Check for painted road markings or full-frame hallucinations
        is_bad, reason = check_false_positive(
            image=orig_img,
            box=(x1, y1, x2, y2),
            class_name=cls_name,
            conf=conf,
            filter_markings=filter_markings
        )

        if is_bad:
            rejected_items.append({
                "class_name": cls_name,
                "conf": conf,
                "reason": reason
            })
            continue

        # Smart Pothole Audit (reclassifies road craters mislabeled as Garbage)
        final_cls, final_conf, final_box, reclass_reason = audit_and_reclassify_detection(
            image=orig_img,
            box=(x1, y1, x2, y2),
            class_name=cls_name,
            conf=conf
        )

        if reclass_reason:
            reclassified_items.append({
                "old_class": cls_name,
                "new_class": final_cls,
                "conf": final_conf,
                "reason": reclass_reason
            })

        boxes_data.append({
            "class_id": cls_id,
            "class_name": final_cls,
            "conf": final_conf,
            "box": final_box
        })

    inference_ms = result.speed.get("inference", 0.0)
    annotated = annotate_image(orig_img, boxes_data, inference_ms, hud=hud)

    # Print summary table in terminal
    display_results_table(
        boxes_data,
        result.speed,
        rejected_items=rejected_items,
        reclassified_items=reclassified_items
    )

    # Handle Saving
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"pred_{Path(image_path).stem}_{int(time.time())}.jpg"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), annotated)
        console.print(f"[bold green]💾 Annotated image successfully saved to:[/bold green] [underline cyan]{save_path}[/underline cyan]")

    # Interactive viewing
    if view:
        win_title = "Garbage Detection AI  [Press 'Q' or ESC to Close | 'S' to Save]"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        
        # Scale window comfortably to user's display
        h, w = annotated.shape[:2]
        max_w, max_h = 1280, 720
        scale = min(max_w / w, max_h / h, 1.0)
        cv2.resizeWindow(win_title, int(w * scale), int(h * scale))
        
        cv2.imshow(win_title, annotated)
        console.print("[dim cyan]Image preview opened. Press 'Q' or ESC in the window to continue, or 'S' to save.[/dim cyan]")
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('s'), ord('S')):
            output_dir.mkdir(parents=True, exist_ok=True)
            quick_save = output_dir / f"saved_{int(time.time())}.jpg"
            cv2.imwrite(str(quick_save), annotated)
            console.print(f"[bold green]💾 Saved screenshot to:[/bold green] {quick_save}")
        cv2.destroyAllWindows()


def process_video_or_stream(
    model: YOLO,
    source: str,
    conf_thresh: float,
    iou_thresh: float,
    filter_markings: bool,
    augment: bool,
    save: bool,
    output_dir: Path,
    view: bool,
    hud: bool
):
    """Runs real-time inference on a video file or webcam stream."""
    src_input = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src_input)

    if not cap.isOpened():
        console.print(f"[bold red]❌ Error: Unable to open video source '{source}'[/bold red]")
        return

    win_title = "Garbage & Hazard Detection - Live Stream [Press 'Q' to Exit]"
    if view:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    writer = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_name = f"pred_video_{int(time.time())}.mp4"
        save_path = output_dir / out_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps_in, (width, height))
        console.print(f"[bold green]📹 Recording video to:[/bold green] [underline cyan]{save_path}[/underline cyan]")

    console.print("[bold green]🚀 Stream started. Press 'q' or ESC in the preview window to exit.[/bold green]")
    
    prev_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps_instant = 1.0 / dt if dt > 0 else 0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant if fps_smooth > 0 else fps_instant

            results = model.predict(
                source=frame,
                conf=conf_thresh,
                iou=iou_thresh,
                augment=augment,
                verbose=False
            )
            result = results[0]

            boxes_data = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = model.names.get(cls_id, f"Class_{cls_id}")

                is_bad, _ = check_false_positive(
                    image=frame,
                    box=(x1, y1, x2, y2),
                    class_name=cls_name,
                    conf=conf,
                    filter_markings=filter_markings
                )
                if is_bad:
                    continue

                final_cls, final_conf, final_box, _ = audit_and_reclassify_detection(
                    image=frame,
                    box=(x1, y1, x2, y2),
                    class_name=cls_name,
                    conf=conf
                )

                boxes_data.append({
                    "class_id": cls_id,
                    "class_name": final_cls,
                    "conf": final_conf,
                    "box": final_box
                })

            inference_ms = result.speed.get("inference", 0.0)
            annotated = annotate_image(frame, boxes_data, inference_ms, hud=hud, fps=fps_smooth)

            if writer:
                writer.write(annotated)

            if view:
                cv2.imshow(win_title, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 27 is ESC
                    console.print("[yellow]Stream stopped by user.[/yellow]")
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Run beautiful YOLO object detection for Garbage and Road Hazards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--source", type=str, default=None, help="Source: Image/video path, '0' for webcam, or URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--augment", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    parser.add_argument("--no-filter", action="store_true", help="Disable road paint marking rejection")
    parser.add_argument("--save", action="store_true", help="Save output with bounding boxes")
    parser.add_argument("--out-dir", type=str, default="runs/predict", help="Output directory for saved results")
    parser.add_argument("--no-view", action="store_true", help="Do not display interactive preview window")
    parser.add_argument("--no-hud", action="store_true", help="Disable top HUD dashboard overlay on images")

    args = parser.parse_args()

    # Resolve Model Path
    resolved_model = resolve_model_path(args.model)
    if not os.path.exists(resolved_model):
        console.print(f"[bold red]❌ Error: Model weights not found at '{resolved_model}'.[/bold red]")
        sys.exit(1)

    # Load Model with Rich Spinner
    with console.status(f"[bold cyan]Loading YOLO Model from [yellow]{resolved_model}[/yellow]...", spinner="dots"):
        model = YOLO(resolved_model)

    classes_str = ", ".join([f"[bold]{name}[/bold]" for name in model.names.values()])
    filter_status = "[bold red]Disabled[/bold red]" if args.no_filter else "[bold green]Active (Road Marking & Pothole Guard)[/bold green]"

    console.print(
        Panel(
            f"📦 [bold cyan]Model Path:[/bold cyan] {resolved_model}\n"
            f"🏷️  [bold cyan]Detected Classes:[/bold cyan] {classes_str}\n"
            f"⚙️  [bold cyan]Confidence Threshold:[/bold cyan] {args.conf:.2f}  │  "
            f"[bold cyan]IoU Threshold:[/bold cyan] {args.iou:.2f}\n"
            f"🛡️  [bold cyan]Road Hazard Guard:[/bold cyan] {filter_status}",
            title="Model & Detection Engine Ready",
            border_style="green"
        )
    )

    # Resolve Source
    source = args.source
    if not source:
        source = select_source_interactive()

    if not source:
        console.print("[yellow]⚠️ No input source selected. Exiting.[/yellow]")
        sys.exit(0)

    output_dir = Path(args.out_dir)
    view_enabled = not args.no_view
    hud_enabled = not args.no_hud
    filter_markings = not args.no_filter

    # Check source type (Webcam / Video vs Image)
    is_webcam = source.isdigit()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    is_video = is_webcam or Path(source).suffix.lower() in video_exts

    if is_video:
        process_video_or_stream(
            model=model,
            source=source,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            filter_markings=filter_markings,
            augment=args.augment,
            save=args.save,
            output_dir=output_dir,
            view=view_enabled,
            hud=hud_enabled
        )
    else:
        process_image(
            model=model,
            image_path=source,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            filter_markings=filter_markings,
            augment=args.augment,
            save=args.save,
            output_dir=output_dir,
            view=view_enabled,
            hud=hud_enabled
        )


if __name__ == "__main__":
    main()
