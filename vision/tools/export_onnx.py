"""Export the YOLO weights to ONNX, which is usually faster on CPU.

PyTorch falls back to slow kernels on CPUs without AVX2/FMA -- the server logs
``Could not initialize NNPACK! Reason: Unsupported hardware``. onnxruntime does not rely
on that path and typically runs the same model considerably faster.

    python tools/export_onnx.py --model yolov8n.pt --imgsz 320

Then point the config at the result: ``PersonTracker(model_path="yolov8n.onnx")``.
Ultralytics loads and tracks with the exported file directly.

Note the export is shape-specific: pass the ``imgsz`` you intend to run at.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="source .pt weights")
    parser.add_argument("--imgsz", type=int, default=320, help="inference resolution to fix")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=10,
        help="frames of random noise to time each variant on (0 to skip)",
    )
    args = parser.parse_args()

    import numpy as np
    from ultralytics import YOLO

    source = YOLO(args.model)
    print(f"exporting {args.model} at imgsz={args.imgsz} ...")
    exported = source.export(format="onnx", imgsz=args.imgsz, simplify=True)
    print(f"wrote {exported}")

    if args.benchmark <= 0:
        return

    image = (np.random.default_rng(0).random((args.imgsz, args.imgsz, 3)) * 255).astype("uint8")
    for label, model in (("pytorch", YOLO(args.model)), ("onnx", YOLO(str(exported)))):
        model.predict(image, imgsz=args.imgsz, verbose=False)  # warm up
        started = time.perf_counter()
        for _ in range(args.benchmark):
            model.predict(image, imgsz=args.imgsz, verbose=False)
        each = (time.perf_counter() - started) / args.benchmark
        print(f"{label:<8} {each * 1000:7.1f} ms per frame")

    print(f"\nto use it, set the tracker model path to {Path(exported).name}")


if __name__ == "__main__":
    main()
