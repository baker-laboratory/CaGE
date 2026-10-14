"""Lay out frames from render_diffusion_trajectory.py as a CFG mechanism panel.

Frames run left to right at one common scale, joined by straight purple arrows; between
the noisy frames, purple arcs above (conditional stream) and grey arcs below
(unconditional); labels underneath. Writes OUT_PREFIX.png (white) and .pdf/.svg
(transparent, arrows and text as vectors).

  python layout_trajectory_panel.py FRAMES_DIR OUT_PREFIX [--gap 0.8]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

PURPLE = "#8C4DE0"
GREY = "#8C8C8C"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("frames_dir", type=Path)
    p.add_argument("out_prefix", type=Path)
    p.add_argument(
        "--gap", type=float, default=0.5, help="space between frames, in frame widths"
    )
    p.add_argument(
        "--labels",
        default=None,
        help="comma-separated labels under frames (mathtext); default: x_{t_n} and "
        "x_{t_{n-1}} under frames 2-3, x_0 under the last",
    )
    p.add_argument(
        "--pad", type=int, default=20, help="px kept around the common content box"
    )
    p.add_argument(
        "--frame_inches", type=float, default=2.0, help="printed width of one frame"
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_frames(frames_dir: Path, pad: int) -> list[Image.Image]:
    """Frames cropped to one shared content box, so they keep a common scale."""
    meta = json.loads((frames_dir / "frames_used.json").read_text())
    images = [Image.open(frames_dir / f["png"]).convert("RGBA") for f in meta]
    boxes = [im.getchannel("A").getbbox() for im in images]
    w, h = images[0].size
    box = (
        max(min(b[0] for b in boxes) - pad, 0),
        max(min(b[1] for b in boxes) - pad, 0),
        min(max(b[2] for b in boxes) + pad, w),
        min(max(b[3] for b in boxes) + pad, h),
    )
    return [im.crop(box) for im in images]


def arrow(ax, start, end, colour: str, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=20,
            color=colour,
            lw=2.8,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main() -> None:
    args = parse_args()
    frames = load_frames(args.frames_dir, args.pad)
    n = len(frames)
    if args.labels is None:
        labels = [""] * n
        labels[-1] = "$x_0$"
        for i, label in ((1, "$x_{t_n}$"), (2, "$x_{t_{n-1}}$")):
            if i < n - 1:
                labels[i] = label
    else:
        labels = args.labels.split(",")
    plt.rcParams["mathtext.fontset"] = "stix"

    # Data units: one frame is 1 wide and h tall
    h = frames[0].height / frames[0].width
    step = 1 + args.gap
    xs = [i * step for i in range(n)]
    rad, inset = 0.35, 0.3  # arc curvature; arcs start/end this far inside the frames
    lift = rad * (step - 2 * inset) / 2  # how far an arc bulges past the frame edge
    x_min, x_max = -1.2 * args.gap, xs[-1] + 1.05
    y_min, y_max = -lift - 0.32, h + lift + 0.08
    fig = plt.figure(
        figsize=(
            (x_max - x_min) * args.frame_inches,
            (y_max - y_min) * args.frame_inches,
        )
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    for x, im in zip(xs, frames):
        ax.imshow(im, extent=(x, x + 1, 0, h), interpolation="lanczos", zorder=2)

    # Straight arrows: into the first frame and between consecutive frames
    margin = 0.15 * args.gap
    arrow(ax, (x_min + 0.1 * args.gap, h / 2), (-margin, h / 2), PURPLE)
    for x in xs[:-1]:
        arrow(ax, (x + 1 + margin, h / 2), (x + step - margin, h / 2), PURPLE)
    # Arcs between the noisy frames only (the last hop into x_0 is a plain arrow)
    for x in xs[:-2]:
        arrow(ax, (x + 1 - inset, h), (x + step + inset, h), PURPLE, rad=-rad)
        arrow(ax, (x + 1 - inset, 0), (x + step + inset, 0), GREY, rad=rad)

    for x, label in zip(xs, labels):
        if label:
            ax.text(
                x + 0.5,
                -lift - 0.12,
                label,
                ha="center",
                va="center",
                fontsize=13 * args.frame_inches,
            )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out_prefix}.png", dpi=args.dpi, facecolor="white")
    for ext in ("pdf", "svg"):
        fig.savefig(f"{args.out_prefix}.{ext}", dpi=args.dpi, transparent=True)


if __name__ == "__main__":
    main()
