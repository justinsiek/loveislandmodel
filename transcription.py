import os
import cv2
from typing import List, Optional


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


if __name__ == "__main__":
    video = os.path.join("videos", "Taylor Ola Initial Chat.mp4")
    sample_frames(video, out_dir="debug_frames", every_sec=1.0, limit=60)