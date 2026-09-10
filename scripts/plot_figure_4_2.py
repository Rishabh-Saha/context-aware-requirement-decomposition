"""Figure 4.2: the Layer 2 file-verification verdict distribution, one stacked bar per condition.

The figure has until now been produced outside the repository, which means the one artefact a
reader looks at first could not be traced to the CSV behind it. This script closes that gap the
same way `export_csv.py` closes it for the tables: it reads
`results/csv/table_4_2_verdict_distribution.csv` and nothing else, recomputes nothing, and draws
exactly the rates that file holds.

Two conventions worth stating, because both were wrong in the earlier figure:

    Class names match Section 3.10.2 and `FileStatus`, so the legend reads Real, Ambiguous,
    Hallucinated. "Verified" was a fifth name for a three-name scheme and invited the reading that
    Layer 2 verifies quality, which it does not: it verifies that a path exists.

    Rates are printed to three decimals. The stored values carry four, and two conditions sum to
    0.9999 there, so rounding to two decimals hides which rows are exact and which are not.

Row sums are checked, never corrected. A row that does not reach 1.000 is reported as a warning,
because the fix for a distribution that does not sum belongs upstream in the verifier, not in the
figure that displays it.

Usage:
    python scripts/plot_figure_4_2.py
    python scripts/plot_figure_4_2.py --csv results/csv/table_4_2_verdict_distribution.csv \
        --out results/figures/figure_4_2_verdict_distribution.png --dpi 300
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DEFAULT_RESULTS_DIR  # noqa: E402

DEFAULT_CSV = f"{DEFAULT_RESULTS_DIR}/csv/table_4_2_verdict_distribution.csv"
DEFAULT_OUT = f"{DEFAULT_RESULTS_DIR}/figures/figure_4_2_verdict_distribution.png"
DEFAULT_DPI = 300

# The three verdicts in the order they stack, left to right, with the CSV column each one reads.
# Hallucinated leads because the bars are read as "how much of this condition's output is not
# grounded", and the legend follows the same order so the key and the bar agree. The display names
# are the class names from Section 3.10.2, not synonyms for them.
CLASSES: tuple[tuple[str, str], ...] = (
    ("hallucinated_rate", "Hallucinated"),
    ("ambiguous_rate", "Ambiguous"),
    ("real_rate", "Real"),
)
# The class the trailing annotation calls out. Named rather than indexed, so reordering the stack
# above cannot silently move the annotation onto a different verdict.
ANNOTATED_COLUMN = "real_rate"

# -- styling ------------------------------------------------------------------------------------
# One block, so matching the figure in the document is a matter of editing these values rather than
# hunting through the drawing code. Everything here matches the surrounding Chapter 4 figures: the
# palette is shared with them, which is why it is a three-hue categorical set rather than one hue
# stepped by grounding quality. Greyscale survival does not depend on the fills anyway, since the
# three differ in lightness and the real rate is carried by the trailing annotation.
SEGMENT_COLORS: tuple[str, str, str] = ("#a4232c", "#f2a882", "#1f6cb0")
# Label ink per segment, chosen against the fill behind it. None means the segment carries no
# inline label: the real segments are the narrow ones, and their value is the trailing annotation.
SEGMENT_LABEL_INK: tuple[str | None, str | None, str | None] = ("white", "black", None)
ANNOTATION_COLOR = "#1f6cb0"
FONT_FAMILY = "serif"
FONT_SERIF = ["DejaVu Serif"]
FONT_SIZE = 13
LABEL_FONT_SIZE = 12
FIGSIZE = (11.5, 5.6)
BAR_HEIGHT = 0.72
TITLE = "Layer 2: where generated file references land"
XLABEL = "Mean proportion of file references"
# The axis runs past 1.0 to leave room for the trailing annotations, which sit outside the bars.
XLIM = (0.0, 1.19)
XTICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Connector from the end of the bar to its annotation: a short rule, then the value.
ANNOTATION_RULE = (1.005, 1.045)
ANNOTATION_X = 1.055
# A segment narrower than this cannot hold "0.000" without spilling into its neighbour, so its
# label is dropped rather than drawn over the next segment. It does not fire on this run's rates:
# the narrowest labelled segment is ambiguous at 0.230.
MIN_LABEL_WIDTH = 0.075
# Condition labels. The raw keys are underscored identifiers built for filenames, and the leave-one-
# out arms are named by what they drop, since the axis has already established that they are full
# RAG variants.
CONDITION_LABELS: dict[str, str] = {
    "vanilla": "No retrieval",
    "full_rag": "Full RAG",
    "full_rag_minus_past_tickets": "minus past tickets",
    "full_rag_minus_design_documents": "minus design documents",
    "full_rag_minus_coding_conventions": "minus coding conventions",
    "full_rag_minus_codebase_summaries": "minus codebase summaries",
}


def read_rows(path: Path) -> list[dict]:
    """The CSV as {condition, n, rates}, in file order.

    File order is the thesis order (baseline, full RAG, then the four leave-one-out arms), so it is
    preserved rather than sorted. Sorting alphabetically would put the ablation arms before the
    baseline they are compared against.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{path} has no rows in it.")

    rows = []
    for row in raw:
        condition = (row.get("condition") or "").strip()
        if not condition:
            continue
        try:
            rates = [float(row[column]) for column, _ in CLASSES]
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path}: {condition} is missing a readable rate ({exc}).") from exc
        rows.append({
            "condition": condition,
            "label": CONDITION_LABELS.get(condition, condition),
            "n": row.get("n"),
            "rates": rates,
        })
    return rows


def sum_report(rows: list[dict], places: int = 3) -> list[str]:
    """Rows whose rates do not sum to 1 at the printed precision, as warning lines.

    The check is done on the rounded values, which is what the figure shows: a row can sum to
    0.9999 in the file and still be exact at three decimals, and it is the printed figure a reader
    would try to add up.
    """
    problems = []
    for row in rows:
        shown = [round(r, places) for r in row["rates"]]
        total = round(sum(shown), places)
        if abs(total - 1.0) >= 10 ** -places:
            problems.append(
                f"{row['condition']}: printed rates sum to {total:.{places}f}, not 1.000 "
                f"(stored sum {sum(row['rates']):.6f})"
            )
    return problems


def draw(rows: list[dict], out_path: Path, dpi: int, places: int = 3) -> Path:
    """Draw and write the figure. Returns the path written."""
    import matplotlib   # lazy, the way the provider SDKs are imported elsewhere
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    annotated = [column for column, _ in CLASSES].index(ANNOTATED_COLUMN)
    # rc_context rather than rcParams.update, so running this inside a larger session does not
    # leave the serif family set for whatever draws next.
    with plt.rc_context({
        "font.family": FONT_FAMILY,
        "font.serif": FONT_SERIF,
        "font.size": FONT_SIZE,
    }):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        positions = list(range(len(rows)))[::-1]   # first CSV row at the top

        # Segment widths are the rounded rates, the same numbers the labels print, so a bar cannot
        # be a fraction of a pixel wider than the value written across it and the three segments
        # add up to exactly the 1.000 the labels claim. The stored four-decimal rates stay
        # untouched on `rows`, which is what the sum check reports against.
        shown = [[round(rate, places) for rate in row["rates"]] for row in rows]

        for index, (_, display) in enumerate(CLASSES):
            ax.barh(
                positions,
                [rates[index] for rates in shown],
                left=[sum(rates[:index]) for rates in shown],
                height=BAR_HEIGHT,
                color=SEGMENT_COLORS[index],
                label=display,
            )

        for rates, y in zip(shown, positions):
            running = 0.0
            for index, rate in enumerate(rates):
                ink = SEGMENT_LABEL_INK[index]
                if ink is not None and rate >= MIN_LABEL_WIDTH:
                    ax.text(
                        running + rate / 2, y, f"{rate:.{places}f}",
                        ha="center", va="center", color=ink, fontsize=LABEL_FONT_SIZE,
                    )
                running += rate
            # The trailing annotation carries the real rate, which is the number the surrounding
            # text argues about and the one whose segment is too narrow to label in place. The rule
            # ties it back to the end of its own bar rather than leaving a floating column of
            # numbers beside six rows.
            ax.plot(list(ANNOTATION_RULE), [y, y], color=ANNOTATION_COLOR, lw=1)
            ax.text(
                ANNOTATION_X, y, f"{rates[annotated]:.{places}f}",
                ha="left", va="center", color=ANNOTATION_COLOR, fontsize=LABEL_FONT_SIZE,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels([row["label"] for row in rows])
        ax.set_xlim(*XLIM)
        ax.set_xticks(list(XTICKS))
        ax.set_xticklabels([f"{t:.2f}" for t in XTICKS])
        ax.set_xlabel(XLABEL, labelpad=10)
        ax.set_title(TITLE, pad=16)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(CLASSES),
            frameon=False, handlelength=1.2, handleheight=1.0, columnspacing=2.5,
        )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Draw Figure 4.2, the Layer 2 verdict distribution per condition."
    )
    p.add_argument("--csv", default=DEFAULT_CSV, help="verdict-distribution CSV to read")
    p.add_argument("--out", default=DEFAULT_OUT, help="where the figure is written")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="output resolution")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}; run scripts/export_csv.py first")

    rows = read_rows(csv_path)
    written = draw(rows, Path(args.out), args.dpi)

    print(f"\nFigure 4.2 from {csv_path}")
    for row in rows:
        rates = "  ".join(f"{display} {row['rates'][i]:.3f}"
                          for i, (_, display) in enumerate(CLASSES))
        print(f"  {row['label']:<38} n={row['n']:<3} {rates}  sum={sum(row['rates']):.3f}")

    problems = sum_report(rows)
    if problems:
        print("\n  WARNING: a verdict distribution does not sum to 1.000 at three decimals.")
        print("  Reported as found; the figure draws the stored rates and corrects nothing.")
        for line in problems:
            print(f"    {line}")
    else:
        print("\n  every row sums to 1.000 at three decimals")

    print(f"\n  written to {written} at {args.dpi} dpi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
