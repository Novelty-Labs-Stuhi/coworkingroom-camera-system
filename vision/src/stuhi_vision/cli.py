"""Command-line interface: run the pipeline, enrol faces, inspect occupancy."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import assembly, config
from .domain import Box, Direction
from .recognition.face import FaceRecognizer
from .recognition.gallery import FaceGallery
from .store import EventStore

app = typer.Typer(help="Single-camera, face-based occupancy tracking for the stuhi office.")

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to the TOML config.")]


def _announce(direction: Direction, name: str | None) -> None:
    if direction is Direction.IN:
        typer.echo(f"[in ]  {name} entered")
    else:
        typer.echo(f"[out]  {name or '(unresolved)'} left")


@app.command()
def run(config_path: ConfigOption = Path("config.toml")) -> None:
    """Process the configured camera/video and track occupancy live."""
    application = assembly.build(config.load(config_path), announce=_announce)
    try:
        application.pipeline.run()
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        application.store.close()


@app.command()
def review(
    output: Annotated[Path, typer.Option("--output", "-o", help="Annotated video path.")] = Path(
        "annotated.mp4"
    ),
    source: Annotated[
        str | None, typer.Option("--source", "-s", help="Override the config source with a file.")
    ] = None,
    config_path: ConfigOption = Path("config.toml"),
) -> None:
    """Run on a recorded clip and write an annotated video (boxes, line, events)."""
    import tempfile
    from dataclasses import replace

    from .config import SourceConfig
    from .visualization import Annotator

    cfg = config.load(config_path)
    if source is not None:
        cfg = replace(cfg, source=SourceConfig(kind="file", target=source))
    # A review run must not touch the real occupancy database.
    review_db = Path(tempfile.gettempdir()) / "stuhi_review.db"
    cfg = replace(cfg, paths=replace(cfg.paths, database=review_db))

    annotator = Annotator(cfg.doorway, output)
    application = assembly.build(cfg, announce=_announce, observer=annotator)
    try:
        application.pipeline.run()
    finally:
        annotator.close()
        application.store.close()
    typer.echo(f"wrote {output}")


@app.command()
def calibrate(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("doorway.png"),
    source: Annotated[str | None, typer.Option("--source", "-s")] = None,
    config_path: ConfigOption = Path("config.toml"),
) -> None:
    """Save one frame with the configured doorway line drawn, to check its placement."""
    import cv2

    from .config import SourceConfig
    from .sources import open_source
    from .visualization import draw_doorway

    cfg = config.load(config_path)
    src = SourceConfig(kind="file", target=source) if source else cfg.source
    frame = next(iter(open_source(src)), None)
    if frame is None:
        typer.echo("no frames from source")
        raise typer.Exit(1)
    image = frame.image.copy()
    draw_doorway(image, cfg.doorway)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    typer.echo(f"wrote {output}  (line {cfg.doorway.line_a} -> {cfg.doorway.line_b})")


@app.command()
def enroll(
    name: str,
    images: list[Path],
    config_path: ConfigOption = Path("config.toml"),
) -> None:
    """Add one person's face(s) to the gallery from one or more photo files."""
    import cv2

    cfg = config.load(config_path)
    gallery = FaceGallery.load(cfg.paths.gallery_dir)
    recognizer = FaceRecognizer(FaceGallery(), cfg.thresholds.face_match)

    added = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            typer.echo(f"  skip (unreadable): {image_path}")
            continue
        height, width = image.shape[:2]
        embedding = recognizer.embed(image, Box(0, 0, width, height))
        if embedding is None:
            typer.echo(f"  skip (no face found): {image_path}")
            continue
        gallery.add(name, embedding)
        added += 1

    gallery.save(cfg.paths.gallery_dir)
    typer.echo(f"enrolled {added} face(s) for {name!r}; gallery now: {gallery.names}")


@app.command()
def occupancy(config_path: ConfigOption = Path("config.toml")) -> None:
    """Print who is currently inside, reconstructed from the event log."""
    cfg = config.load(config_path)
    store = EventStore(cfg.paths.database)
    try:
        inside = _replay(store.recent(limit=10_000))
    finally:
        store.close()
    if not inside:
        typer.echo("nobody inside")
        return
    typer.echo(f"{len(inside)} inside: {', '.join(sorted(inside))}")


def _replay(events_newest_first: list[tuple[float, str, str]]) -> set[str]:
    inside: set[str] = set()
    for _ts, name, direction in reversed(events_newest_first):
        if direction == Direction.IN.value:
            inside.add(name)
        else:
            inside.discard(name)
    return inside
