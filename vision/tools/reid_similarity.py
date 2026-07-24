"""Offline person re-ID over recorded clips.

Built bottom-up from four pieces:
  1. detect_person(frame)          -- find the person in one frame, return their crop
  2. embed_image(crop)             -- turn one crop into an appearance vector (OSNet)
  3. embed_video(path)             -- detect+embed every frame, return the average vector
  4. main()                        -- embed all clips, build the similarity matrix, plot it

    pip install torchreid matplotlib
    python vision/tools/reid_similarity.py --clips clips/ --out reid_matrix.png

Caveat: the clips are grayscale QVGA, well below what re-ID models were trained on, so
scores are a coarse ranking (same person tends higher), not a hard identity verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the vision engine importable whether or not it was pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stuhi_vision.detection import PersonDetector  # noqa: E402
from stuhi_vision.recognition.embeddings import normalize  # noqa: E402


# --- 1) Detect the person in a single frame ---------------------------------
def detect_person(frame: np.ndarray, detector: PersonDetector) -> np.ndarray | None:
    """Return the largest person crop in ``frame`` as an RGB array, or None.

    ``frame`` is a BGR image (as OpenCV reads it). We pick the largest detected box
    -- the closest / most complete view of the person -- and return that crop
    converted to RGB, which is what the embedder expects.
    """
    boxes = detector.detect(frame)
    if not boxes:
        return None
    box = max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
    crop = box.crop(frame)
    if crop.size == 0:
        return None
    return crop[:, :, ::-1].copy()  # BGR -> RGB


# --- 2) Embed a single picture ----------------------------------------------
def embed_image(image_rgb: np.ndarray, extractor) -> np.ndarray:
    """Turn one RGB crop into an L2-normalised appearance vector."""
    features = extractor([image_rgb]).cpu().numpy()  # (1, dim)
    return normalize(features[0].astype(np.float32))


# --- 3) Average embedding over a whole video --------------------------------
def embed_video(
    video: Path, detector: PersonDetector, extractor
) -> np.ndarray | None:
    """Detect + embed the person in every frame, return the averaged vector.

    Returns None if no frame contained a detectable person.
    """
    cap = cv2.VideoCapture(str(video))
    vectors: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = detect_person(frame, detector)
        if crop is not None:
            vectors.append(embed_image(crop, extractor))
    cap.release()

    if not vectors:
        return None
    return normalize(np.mean(vectors, axis=0).astype(np.float32))


# --- 4) Matrix + plotting + CLI ---------------------------------------------
def build_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    """Cosine-similarity matrix of already-normalised vectors: E @ E.T."""
    matrix = np.vstack(embeddings)
    return matrix @ matrix.T


def render(sim: np.ndarray, labels: list[str], out: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.6)))
    im = ax.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{sim[i, j]:.2f}",
                ha="center", va="center", fontsize=6,
                color="white" if sim[i, j] < 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("Clip-to-clip person re-ID similarity")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote heatmap -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("clips"), help="clips dir")
    parser.add_argument("--out", type=Path, default=Path("reid_matrix.png"))
    parser.add_argument("--model", default="osnet_x1_0", help="torchreid model name")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    videos = sorted(args.clips.glob("*.mp4"))
    if not videos:
        sys.exit(f"No .mp4 clips found in {args.clips}")

    try:
        # Package layout moved between torchreid releases.
        try:
            from torchreid.reid.utils import FeatureExtractor
        except ModuleNotFoundError:
            from torchreid.utils import FeatureExtractor
    except ModuleNotFoundError as exc:
        sys.exit(f"Could not import torchreid FeatureExtractor: {exc}")

    extractor = FeatureExtractor(model_name=args.model, model_path="", device=args.device)
    detector = PersonDetector()

    labels: list[str] = []
    embeddings: list[np.ndarray] = []
    for video in videos:
        vec = embed_video(video, detector, extractor)
        if vec is None:
            print(f"  {video.name}: no person detected -- skipped")
            continue
        labels.append(video.stem.replace("2026-", ""))  # trim year for readable labels
        embeddings.append(vec)
        print(f"  {video.name}: embedded")

    if len(embeddings) < 2:
        sys.exit("Need at least 2 clips with a detected person to compare.")

    sim = build_matrix(embeddings)

    print("\nSimilarity matrix (rows/cols = clips, in order):")
    for name, row in zip(labels, sim, strict=True):
        print(f"  {name:>16}  " + " ".join(f"{v:.2f}" for v in row))

    csv = args.out.with_suffix(".csv")
    lines = ["clip," + ",".join(labels)] + [
        f"{name}," + ",".join(f"{v:.4f}" for v in row)
        for name, row in zip(labels, sim, strict=True)
    ]
    csv.write_text("\n".join(lines) + "\n")
    print(f"Wrote CSV -> {csv}")

    render(sim, labels, args.out)


if __name__ == "__main__":
    main()
