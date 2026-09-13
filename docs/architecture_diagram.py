"""Renders docs/architecture.png (matplotlib, no external diagram tool). Source of record for the
architecture diagram embedded in README.md. Regenerate with:

    .venv/bin/python docs/architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties

OUT_PATH = Path(__file__).parent / "architecture.png"

BLUE = "#3b6ea5"
GREEN = "#4a8b5c"
ORANGE = "#c07a2b"
GREY = "#6b6b6b"
PURPLE = "#7a5ba6"
RED = "#8a3b3b"

bold = FontProperties(weight="bold")


def box(ax, xy, w, h, text, color, fontsize=9, text_color="white"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=color, alpha=0.94, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=3, linespacing=1.35)


def arrow(ax, start, end, color=GREY, lw=1.5, connectionstyle="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        color=color, lw=lw, zorder=1, connectionstyle=connectionstyle,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 9))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(6.25, 8.72, "Context-aware requirements decomposition: pipeline",
            ha="center", fontsize=13, fontproperties=bold)

    # Row 1: four source types feeding one ingestion/indexing box
    ax.text(1.55, 8.05, "Sources", ha="center", fontsize=9.5, fontproperties=bold, color=GREY)
    sources = [
        ("Past tickets\n(title + description text;\nissue key is metadata only)", 7.15),
        ("Design documents\n(Forrest xdocs prose)", 6.15),
        ("Coding conventions\n(checkstyle.xml, README)", 5.15),
        ("Codebase summaries\n(one LLM summary per file)", 4.15),
    ]
    for label, y in sources:
        box(ax, (0.1, y), 3.0, 0.8, label, GREY, fontsize=7.8)

    box(ax, (3.6, 5.15), 2.15, 2.4, "Ingestion &\nindexing\n\nchunk (500/50),\nembed, store per\ncontext type in a\nshared vector index",
        BLUE, fontsize=8.2)
    for _, y in sources:
        arrow(ax, (3.1, y + 0.4), (3.6, 6.35))

    # Requirement input, bottom left
    box(ax, (0.1, 2.5), 3.0, 0.8, "Requirement in\n(Jira issue title +\ndescription)", ORANGE, fontsize=8.5)

    # Retrieval block: one representative per-type lane, repeated per active context type
    ax.text(6.85, 8.05, "Retrieval — repeated per active context type, budget 2 chunks each",
            ha="center", fontsize=9, fontproperties=bold, color=GREY)
    box(ax, (5.9, 6.9), 1.7, 0.95, "Dense\nquery\n(embedding\nsimilarity)", BLUE, fontsize=7.6)
    box(ax, (5.9, 5.6), 1.7, 0.95, "Lexical\nquery\n(token-overlap\nset intersection)", BLUE, fontsize=7.3)
    box(ax, (8.1, 6.25), 2.0, 0.95, "Reciprocal rank\nfusion, k = 50\n(within this type)", PURPLE, fontsize=8)

    arrow(ax, (7.6, 7.3), (8.1, 6.9))
    arrow(ax, (7.6, 6.0), (8.1, 6.6))
    arrow(ax, (4.0, 6.5), (5.9, 6.9), color=BLUE, lw=1.1)
    arrow(ax, (4.0, 5.9), (5.9, 6.0), color=BLUE, lw=1.1)

    # Requirement -> retrieval lanes, routed clear of the ingestion box (right edge x=5.75)
    ax.plot([3.1, 6.75], [2.9, 2.9], color=GREY, lw=1.3, zorder=1)
    ax.plot([6.75, 6.75], [2.9, 6.72], color=GREY, lw=1.3, zorder=1)
    arrow(ax, (6.75, 6.72), (5.95, 7.28))
    arrow(ax, (6.75, 6.72), (5.95, 6.05))
    ax.text(6.85, 5.02,
            "Runs once per active context type.\nVanilla activates none; leave-one-out omits exactly one.",
            ha="center", fontsize=7.2, color=GREY, style="italic")

    # Prompt assembly under budget
    box(ax, (9.2, 4.5), 2.9, 1.45,
        "Prompt assembly under budget\n\nOne labelled section per\nactive context type, requirement\nappended last. Ablation DROPS\na section — the prompt gets\nshorter, nothing is redistributed.",
        ORANGE, fontsize=7.6)
    arrow(ax, (9.1, 6.7), (9.6, 5.95), connectionstyle="arc3,rad=0.15")

    # Generation
    box(ax, (9.2, 2.9), 2.9, 1.0, "Generation\ngpt-4o, temperature 0.0\n-> fixed JSON decomposition",
        GREEN, fontsize=8.3)
    arrow(ax, (10.65, 4.5), (10.65, 3.9))

    # Evaluation layers hanging off the output
    ax.text(6.25, 2.15, "Evaluation — every layer runs on the same generated decomposition",
            ha="center", fontsize=9.2, fontproperties=bold, color=GREY)
    layers = [
        ("Layer 1\nStructural metrics\n(diagnostic)", 0.1),
        ("Layer 2\nFile-existence check\nREAL / AMBIGUOUS /\nHALLUCINATED", 3.15),
        ("Layer 3\nPairwise LLM judge\ncross-family, descriptive\nunless Layer 4 gate passes", 6.2),
        ("Layer 4\nHuman calibration\nkappa vs. judge,\ngates Layer 3", 9.25),
    ]
    for label, x in layers:
        box(ax, (x, 0.15), 3.0, 1.35, label, RED, fontsize=7.6)
        arrow(ax, (10.65, 2.9), (x + 1.5, 1.5), color=RED, lw=1.0)

    fig.savefig(OUT_PATH, dpi=170, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
