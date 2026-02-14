#!/usr/bin/env python3
"""
Generate resource reproducibility flow figure for manuscript packaging.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [
    PROJECT_ROOT / "figures" / "resource_reproducibility_flow.png",
    PROJECT_ROOT / "paper" / "figures" / "resource_reproducibility_flow.png",
]


def draw_box(ax, x, y, w, h, title, detail, color):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + w / 2,
        y + h * 0.33,
        detail,
        ha="center",
        va="center",
        fontsize=9,
        color="#1f2937",
    )


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "#111827"},
    )


def generate():
    fig, ax = plt.subplots(figsize=(13, 4.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(
        ax,
        0.03,
        0.3,
        0.17,
        0.4,
        "Input Snapshot",
        "atlas_resource/*\nmanifest inputs + hashes",
        "#dbeafe",
    )
    draw_box(
        ax,
        0.24,
        0.3,
        0.17,
        0.4,
        "Database Build",
        "schema.sql + build_db.py\nrelease/hpcma_atlas.db",
        "#dcfce7",
    )
    draw_box(
        ax,
        0.45,
        0.3,
        0.17,
        0.4,
        "Query Contracts",
        "database/queries/*.sql\ntest_query_regression.py",
        "#fef3c7",
    )
    draw_box(
        ax,
        0.66,
        0.3,
        0.17,
        0.4,
        "Reproducibility Gate",
        "run_all.sh + pytest\ndata/schema/claim checks",
        "#fee2e2",
    )
    draw_box(
        ax,
        0.86,
        0.3,
        0.11,
        0.4,
        "Paper Artifacts",
        "methods + figures\nreviewer-auditable",
        "#ede9fe",
    )

    arrow(ax, 0.20, 0.5, 0.24, 0.5)
    arrow(ax, 0.41, 0.5, 0.45, 0.5)
    arrow(ax, 0.62, 0.5, 0.66, 0.5)
    arrow(ax, 0.83, 0.5, 0.86, 0.5)

    ax.set_title(
        "HPCMA Resource Reproducibility Flow",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        pad=12,
    )

    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved: {output}")

    plt.close(fig)


if __name__ == "__main__":
    generate()
