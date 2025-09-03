import os
import cv2
from typing import List, Optional
import numpy as np
import re
from PIL import Image
import pytesseract
from datetime import timedelta


def sample_frames(
    video_path: str,
    out_dir: str = "debug_frames",
    every_sec: float = 1.0,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Save a frame every `every_sec` seconds to `out_dir`.
    Returns a list of saved file paths.
    """
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * every_sec)))

    saved: List[str] = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        path = os.path.join(out_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(path, frame)
        saved.append(path)
        if limit and len(saved) >= limit:
            break

    cap.release()
    print(f"Saved {len(saved)} frames to '{out_dir}'")
    return saved


def find_subtitle_polygon(
    frame_bgr,
    white_thresh: int = 250,
    expand_px: int = 10,
):
    """
    Detect a white subtitle box (black text on white) and return a straight-edged polygon.
    - Thresholds for white.
    - Morphologically closes to fill text holes and connects lines.
    - Optionally dilates outward by `expand_px` to avoid clipping near edges.
    - Chooses the largest region, then approximates to straight lines.
    Returns: contour (Nx1x2) in full-frame coordinates, or None.
    """
    H, W = frame_bgr.shape[:2]
    y0 = 0
    roi = frame_bgr[y0:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)), iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 11)), iterations=1)
    if expand_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * expand_px + 1, 2 * expand_px + 1))
        mask = cv2.dilate(mask, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    best = best + np.array([0, y0]).reshape(1, 1, 2)

    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.004 * peri, True)
    return approx


def draw_polygons_on_frames(
    frames_dir: str = "debug_frames",
    out_dir: str = "debug_sub_overlays",
    white_thresh: int = 250,
    expand_px: int = 10,
) -> int:
    """
    For each saved frame, draw the detected subtitle polygon (if any) and save overlay.
    Returns the number of overlays saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([n for n in os.listdir(frames_dir) if n.lower().endswith((".jpg", ".png"))])

    count = 0
    for name in names:
        path = os.path.join(frames_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            continue

        poly = find_subtitle_polygon(frame, white_thresh=white_thresh, expand_px=expand_px)
        overlay = frame.copy()
        if poly is not None:
            cv2.drawContours(overlay, [poly], -1, (0, 255, 0), 3)
        else:
            cv2.putText(overlay, "NO SUB POLYGON", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_dir, f"overlay_{name}"), overlay)
        count += 1

    print(f"Saved {count} overlays to '{out_dir}'")
    return count


def to_text_only(
    crop_bgr,
    mask_roi,
    text_thresh: int = 0,
    dilate_px: int = 1,
    border_clear_px: int = 20,   # distance (px) we must be inside the polygon to be kept
):
    # Compute "interior-only" mask via distance transform
    mask_u8 = (mask_roi > 0).astype(np.uint8) * 255
    if border_clear_px > 0:
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        keep_core = (dist > float(border_clear_px)).astype(np.uint8) * 255
    else:
        keep_core = mask_u8

    # Detect dark text
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if text_thresh <= 0:
        _, text = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, text = cv2.threshold(gray, text_thresh, 255, cv2.THRESH_BINARY_INV)

    # Light cleanup
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_px + 1, 2 * dilate_px + 1))
        text = cv2.morphologyEx(text, cv2.MORPH_OPEN, k, iterations=1)

    # Remove anything too close to the polygon edge (kills the black border)
    text = cv2.bitwise_and(text, keep_core)

    # Compose white background with black text
    out = np.full_like(crop_bgr, 255)
    out[text > 0] = (0, 0, 0)
    return out


def crop_polygons_on_frames(
    frames_dir: str = "debug_frames",
    out_dir: str = "debug_sub_crops",
    white_thresh: int = 245,
    pad_px: int = 6,
    erode_mask_px: int = 2,
    whiten_outside: bool = True,
    text_only: bool = True,
    text_thresh: int = 0,
    dilate_px: int = 1,
) -> int:
    """
    - Finds the subtitle polygon.
    - Crops its bounding rect with small padding.
    - Optionally whitens outside the polygon.
    - Optionally outputs text-only (white background, black text) inside the polygon.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([n for n in os.listdir(frames_dir) if n.lower().endswith((".jpg", ".png"))])

    count = 0
    for name in names:
        path = os.path.join(frames_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            continue

        poly = find_subtitle_polygon(frame, white_thresh=white_thresh)
        if poly is None:
            continue

        x, y, w, h = cv2.boundingRect(poly)
        H, W = frame.shape[:2]
        x = max(0, x - pad_px)
        y = max(0, y - pad_px)
        w = min(W - x, w + 2 * pad_px)
        h = min(H - y, h + 2 * pad_px)

        crop = frame[y:y + h, x:x + w].copy()

        mask_full = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask_full, [poly], -1, 255, thickness=-1)
        mask_roi = mask_full[y:y + h, x:x + w]

        if erode_mask_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_mask_px + 1, 2 * erode_mask_px + 1))
            mask_roi = cv2.erode(mask_roi, k, iterations=1)

        if whiten_outside:
            crop[mask_roi == 0] = 255

        if text_only:
            crop = to_text_only(
                crop, mask_roi,
                text_thresh=0,
                dilate_px=1,
                border_clear_px=20, 
            )

        out_path = os.path.join(out_dir, f"sub_{name}")
        cv2.imwrite(out_path, crop)
        count += 1

    print(f"Saved {count} crops to '{out_dir}'")
    return count


def configure_tesseract(explicit_path: Optional[str] = None) -> None:
    # Set the path to tesseract.exe if needed
    if explicit_path:
        pytesseract.pytesseract.tesseract_cmd = explicit_path
        return
    guesses = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for g in guesses:
        if os.path.exists(g):
            pytesseract.pytesseract.tesseract_cmd = g
            break

def _parse_frame_index(name: str) -> Optional[int]:
    m = re.search(r"(\d+)(?=\.[A-Za-z]+$)", name)
    return int(m.group(1)) if m else None


def _get_fps(video_path: Optional[str]) -> Optional[float]:
    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps if fps > 0 else None


def _ocr_image(img_bgr) -> str:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    txt = pytesseract.image_to_string(pil, config="--oem 3 --psm 6", lang="eng")
    return " ".join(txt.replace("|", "I").split()).strip()


def ocr_crops_to_transcript(
    crops_dir: str = "debug_sub_crops",
    video_path: Optional[str] = None,
    out_csv_path: str = "ocr_per_frame.csv",
    out_text_path: str = "transcript_step3.txt",
) -> str:
    names = sorted([n for n in os.listdir(crops_dir) if n.lower().endswith((".jpg", ".png"))])
    if not names:
        print(f"No crops found in '{crops_dir}'")
        return ""

    fps = _get_fps(video_path)
    rows = []
    transcript_lines: List[str] = []
    last_text = ""

    for name in names:
        path = os.path.join(crops_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue

        text = _ocr_image(img)
        if not text:
            continue

        frame_idx = _parse_frame_index(name) or -1
        ts = ""
        if fps and frame_idx >= 0:
            sec = frame_idx / fps
            ts = str(timedelta(seconds=sec)).split(".")[0]

        rows.append([name, frame_idx, ts, text])

        if text != last_text:
            transcript_lines.append(text)
            last_text = text

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["file", "frame_index", "timestamp", "text"])
        w.writerows(rows)

    transcript = "\n".join(transcript_lines)
    with open(out_text_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"Wrote {len(rows)} rows to '{out_csv_path}' and {len(transcript_lines)} lines to '{out_text_path}'")
    return transcript


if __name__ == "__main__":
    configure_tesseract() 
    sample_frames(video_path="videos/Taylor Ola Initial Chat.mp4", out_dir="debug_frames", every_sec=1.0, limit=60)
    draw_polygons_on_frames(frames_dir="debug_frames", out_dir="debug_sub_overlays", white_thresh=245)
    crop_polygons_on_frames(
        frames_dir="debug_frames",
        out_dir="debug_sub_crops",
        white_thresh=250,
        pad_px=6,
        erode_mask_px=2,
        whiten_outside=True,
        text_only=True,
        text_thresh=0,
        dilate_px=1,
    )

    video = os.path.join("videos", "Taylor Ola Initial Chat.mp4")
    ocr_crops_to_transcript(
        crops_dir="debug_sub_crops",
        video_path=video,
        out_csv_path="ocr_per_frame.csv",
        out_text_path="transcript_step3.txt",
    )