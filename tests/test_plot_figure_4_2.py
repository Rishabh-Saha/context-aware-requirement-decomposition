"""Tests for scripts/plot_figure_4_2.py.

The drawing itself is not worth asserting on, but two things around it are. The figure must show
the CSV's row order rather than an alphabetical one, because alphabetical puts the ablation arms
above the baseline they are compared against. And the sum check must fire on the printed precision,
not the stored one: a row storing 0.9999 is exact at three decimals and must not warn, while a row
that genuinely does not sum must warn rather than being quietly normalised.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_figure_4_2 as pf  # noqa: E402

HEADER = ["condition", "n", "real_rate", "ambiguous_rate", "hallucinated_rate"]


def write_csv(path: Path, rows: list[list]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def test_row_order_follows_the_file_not_the_alphabet(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["vanilla", 20, 0.0667, 0.23, 0.7033],
        ["full_rag", 20, 0.0583, 0.4458, 0.4958],
        ["full_rag_minus_past_tickets", 20, 0.0458, 0.3983, 0.5558],
    ])
    rows = pf.read_rows(path)

    assert [r["condition"] for r in rows] == [
        "vanilla", "full_rag", "full_rag_minus_past_tickets"
    ]
    # Rates come back in stacking order, hallucinated first, not in CSV column order.
    assert rows[0]["rates"] == [0.7033, 0.23, 0.0667]
    assert rows[1]["label"] == "Full RAG", "condition keys are spelled out for the axis"


def test_a_row_storing_four_decimals_is_exact_at_three(tmp_path):
    """full_rag stores 0.9999. That is rounding in the stored rates, not a missing verdict, and it
    reaches 1.000 at the precision the figure prints, so it must not be reported as a problem."""
    path = write_csv(tmp_path / "t.csv", [["full_rag", 20, 0.0583, 0.4458, 0.4958]])
    rows = pf.read_rows(path)

    assert sum(rows[0]["rates"]) < 1.0
    assert pf.sum_report(rows) == []


def test_a_row_that_does_not_sum_is_reported_rather_than_adjusted(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["vanilla", 20, 0.0667, 0.23, 0.7033],
        ["full_rag", 20, 0.05, 0.40, 0.45],
    ])
    rows = pf.read_rows(path)
    problems = pf.sum_report(rows)

    assert len(problems) == 1
    assert "full_rag" in problems[0] and "0.900" in problems[0]
    # The rates themselves are untouched, so the figure draws what the CSV holds.
    assert rows[1]["rates"] == [0.45, 0.40, 0.05]


def test_a_missing_rate_column_raises_rather_than_drawing_a_gap(tmp_path):
    path = tmp_path / "t.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n", "real_rate", "ambiguous_rate"])
        writer.writerow(["vanilla", 20, 0.0667, 0.23])

    try:
        pf.read_rows(path)
    except SystemExit as exc:
        assert "vanilla" in str(exc)
    else:
        raise AssertionError("a missing verdict column must raise, not draw a short bar")


def test_the_figure_is_written_at_the_requested_dpi(tmp_path):
    """300 dpi is the submission requirement, so it is checked against the file rather than trusted
    to the call. PNG carries its resolution in the pHYs chunk, in pixels per metre."""
    from PIL import Image

    path = write_csv(tmp_path / "t.csv", [
        ["vanilla", 20, 0.0667, 0.23, 0.7033],
        ["full_rag", 20, 0.0583, 0.4458, 0.4958],
    ])
    out = pf.draw(pf.read_rows(path), tmp_path / "figures" / "fig.png", dpi=300)

    assert out.exists()
    with Image.open(out) as img:
        dpi_x, dpi_y = img.info["dpi"]
    assert round(dpi_x) == 300 and round(dpi_y) == 300
