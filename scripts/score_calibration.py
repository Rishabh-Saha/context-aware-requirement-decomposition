"""Layer 4 step two: score researcher-versus-judge agreement and apply the pre-registered gate.

Reads the filled calibration_ratings.csv alongside key.json, and reports Cohen's kappa between the
two raters overall and per criterion.

The researcher's answers are sheet positions (A or B on a randomised sheet) while the judge's are
side names (a condition, or the reference sentinel). Those are not comparable until both are
expressed the same way, so every answer is translated through key.json. Comparing raw letters
without that translation would silently score the randomisation rather than the agreement.

Which of the two vocabularies kappa is then computed in matters, and the choice here is the binary
one. Both answers are reduced to the sheet position, so the label space is {A, B} and chance
agreement lands near 0.5, which is the honest baseline for a forced binary choice on a randomised
sheet. Running kappa over side names instead would spread the marginals across seven labels, drive
expected agreement towards zero, and inflate kappa towards raw agreement. Each pair's side names are
still recorded per pair in result.json, so the translation stays auditable.

Cells are dropped, never imputed. A blank rating is an absent measurement, and so is a criterion
where the judge's two presentation orders disagreed and reconciled to null. Filling either with a
guess would inflate agreement, so each kappa reports its own n.

The gate is on the overall winner, since the reconciled winner is what Layer 3 contributes to the
analysis. Per-criterion and pooled figures are reported alongside as supporting detail.

Usage:
    python scripts/score_calibration.py --run-id 20260814T033139Z
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.calibration import (  # noqa: E402
    KAPPA_THRESHOLD,
    OVERALL,
    RATING_COLUMNS,
    calibrate,
)
from src.eval.judge import CRITERIA  # noqa: E402
from src.utils.io import write_json  # noqa: E402

from src.paths import CALIBRATION_SUBDIR, DEFAULT_RUNS_DIR  # noqa: E402

# OVERALL and RATING_COLUMNS come from src.eval.calibration, the same place
# build_calibration_set.py reads them, so this reader cannot drift from that writer.

BLANK_MARKERS = {"", "-", "na", "n/a", "skip"}


def read_ratings(path: Path) -> dict[str, dict[str, str | None]]:
    """The researcher's sheet answers as {pair_id: {column: "A"/"B"/None}}.

    Anything that is neither a blank marker nor A or B raises. A typo silently read as a skip would
    shrink n without saying so, and read as a choice would fabricate a rating.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    ratings: dict[str, dict[str, str | None]] = {}
    problems: list[str] = []
    for row in rows:
        pair_id = (row.get("pair_id") or "").strip()
        if not pair_id:
            continue
        answers: dict[str, str | None] = {}
        for column in RATING_COLUMNS:
            raw = (row.get(column) or "").strip()
            if raw.lower() in BLANK_MARKERS:
                answers[column] = None
                continue
            choice = raw.upper()
            if choice not in ("A", "B"):
                problems.append(f"{pair_id}.{column} = {raw!r}")
                continue
            answers[column] = choice
        ratings[pair_id] = answers

    if problems:
        raise SystemExit("unreadable ratings (expected A, B or blank):\n  " + "\n  ".join(problems))
    return ratings


def researcher_side(choice: str | None, key_row: dict) -> str | None:
    """Translate a sheet position into the side it names, which is the vocabulary the judge used."""
    if choice is None:
        return None
    return key_row["sheet_a"] if choice == "A" else key_row["sheet_b"]


def judge_position(side: str | None, key_row: dict) -> str | None:
    """Translate a judge side name into the position it occupies on this rater's sheet.

    A side that is neither sheet_a nor sheet_b means the key and the judgment have drifted apart,
    which would quietly corrupt every downstream number, so it raises rather than returning a guess.
    """
    if side is None:
        return None
    if side == key_row["sheet_a"]:
        return "A"
    if side == key_row["sheet_b"]:
        return "B"
    raise SystemExit(
        f"{key_row['pair_id']}: judge chose {side!r}, which is neither side on the sheet "
        f"({key_row['sheet_a']!r}, {key_row['sheet_b']!r}). The key and the judgments disagree."
    )


def paired_choices(ratings: dict, keys: dict, column: str) -> tuple[list[str], list[str], list[dict]]:
    """Aligned researcher and judge positions for one column, plus a record of each usable pair.

    Both raters come out in sheet-position space, reached from opposite directions: the researcher's
    letter is checked against the side it names, and the judge's side name is mapped to a letter.
    The per-pair record keeps both vocabularies so the translation can be audited afterwards.
    """
    researcher: list[str] = []
    judge: list[str] = []
    used: list[dict] = []
    for pair_id in sorted(keys):
        key_row = keys[pair_id]
        judge_side = (key_row["judge_winner"] if column == OVERALL
                      else key_row["judge_per_criterion"].get(column))
        rater_letter = ratings.get(pair_id, {}).get(column)
        rater_side = researcher_side(rater_letter, key_row)
        judge_letter = judge_position(judge_side, key_row)
        if judge_letter is None or rater_letter is None:
            continue
        researcher.append(rater_letter)
        judge.append(judge_letter)
        used.append({
            "pair_id": pair_id,
            "researcher_position": rater_letter,
            "researcher_side": rater_side,
            "judge_position": judge_letter,
            "judge_side": judge_side,
            "agreed": rater_letter == judge_letter,
        })
    return researcher, judge, used


def kappa_block(researcher: list[str], judge: list[str], gate_eligible: bool = False) -> dict:
    """calibrate() plus the guards json and a human reader both need.

    Two degenerate cases are handled rather than allowed to propagate. With fewer than two usable
    pairs there is nothing to compute. With perfect agreement on a single label, chance agreement is
    1 and Cohen's kappa is undefined; scikit-learn returns nan, which is not valid JSON and would
    otherwise fail the gate comparison silently rather than saying why.

    `layer3_validated` survives only on the gate-eligible block. calibrate() attaches it to every
    block it scores, but the pre-registered gate is defined on the overall winner alone, so a pooled
    or per-criterion kappa above 0.6 carrying `layer3_validated: true` reads as a pass that the
    design never grants. The kappa and the threshold both stay on every block, so a reader can still
    see where each one sits relative to 0.6 without that comparison claiming to be the gate.
    """
    n = len(researcher)
    if n < 2:
        return {"kappa": None, "n": n, "usable": False,
                "note": "fewer than two usable pairs, kappa is not defined"}

    result = calibrate(researcher, judge)
    kappa = result["kappa"]
    if math.isnan(kappa):
        degenerate = {
            "kappa": None, "n": n, "usable": False,
            "raw_agreement": round(sum(a == b for a, b in zip(researcher, judge)) / n, 4),
            "threshold": KAPPA_THRESHOLD,
            "note": ("kappa is undefined because one rater used a single label throughout, so "
                     "chance agreement is 1. Raw agreement is reported instead."),
        }
        if gate_eligible:
            degenerate["layer3_validated"] = False
        return degenerate

    if not gate_eligible:
        result.pop("layer3_validated", None)

    # cohens_kappa always attaches ci95 once n >= 2, which the guard above has established.
    result["ci95"] = [None if math.isnan(v) else v for v in result["ci95"]]
    result["usable"] = True
    result["raw_agreement"] = round(sum(a == b for a, b in zip(researcher, judge)) / n, 4)
    return result


def format_line(label: str, block: dict, gate_eligible: bool = False) -> str:
    """One printed line per scope. PASS or FAIL is only for the gate-eligible scope.

    The gate is the overall winner, so a pooled or per-criterion line reading PASS would announce an
    outcome the design does not grant, the same misreading `layer3_validated` used to invite in the
    JSON. Those lines carry the threshold instead, which lets a reader place the kappa against 0.6
    without the placement being dressed up as a verdict.
    """
    if not block.get("usable"):
        return f"  {label:<20} n={block['n']:<3} kappa n/a  ({block.get('note', '')})"
    ci = block.get("ci95") or (None, None)
    ci_text = f"[{ci[0]}, {ci[1]}]" if ci[0] is not None else "[n/a]"
    if gate_eligible:
        closing = "PASS" if block["kappa"] >= KAPPA_THRESHOLD else "FAIL"
    else:
        closing = f"threshold {KAPPA_THRESHOLD}"
    return (f"  {label:<20} n={block['n']:<3} kappa={block['kappa']:<7} 95% CI {ci_text:<18} "
            f"agreement={block['raw_agreement']:<6} {closing}")


def main() -> int:
    p = argparse.ArgumentParser(description="Score Layer 4 researcher-versus-judge agreement.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    # The published results package carries the ratings and the key under different names from the
    # run directory, so these two overrides let a reader recompute the gate from the package alone
    # rather than needing the full run tree. Without them Layer 4 would be the one reported figure
    # that has to be taken on trust, which is the wrong figure to make unverifiable.
    p.add_argument("--ratings", default=None,
                   help="path to the filled ratings CSV; defaults to the run's calibration directory")
    p.add_argument("--key", default=None,
                   help="path to the hidden key; defaults to the run's calibration directory")
    p.add_argument("--out", default=None,
                   help="where to write result.json; defaults to the run's calibration directory")
    args = p.parse_args()

    cal_dir = Path(args.runs_dir) / args.run_id / CALIBRATION_SUBDIR
    ratings_path = Path(args.ratings) if args.ratings else cal_dir / "calibration_ratings.csv"
    key_path = Path(args.key) if args.key else cal_dir / "key.json"
    out_path = Path(args.out) if args.out else cal_dir / "result.json"
    for path in (ratings_path, key_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/build_calibration_set.py first")

    keys = json.loads(key_path.read_text())
    ratings = read_ratings(ratings_path)

    unrated = [pid for pid in sorted(keys)
               if all(v is None for v in ratings.get(pid, {}).values())]
    if len(unrated) == len(keys):
        raise SystemExit(f"{ratings_path} has no ratings in it yet.")

    per_criterion = {}
    pooled_researcher: list[str] = []
    pooled_judge: list[str] = []
    for criterion in CRITERIA:
        researcher, judge, _ = paired_choices(ratings, keys, criterion)
        per_criterion[criterion] = kappa_block(researcher, judge)
        pooled_researcher += researcher
        pooled_judge += judge

    overall_researcher, overall_judge, overall_pairs = paired_choices(ratings, keys, OVERALL)
    overall = kappa_block(overall_researcher, overall_judge, gate_eligible=True)
    pooled = kappa_block(pooled_researcher, pooled_judge)

    gate_pass = bool(overall.get("usable") and overall["kappa"] >= KAPPA_THRESHOLD)

    print(f"\nLayer 4 calibration for {args.run_id}")
    print(f"  threshold: kappa >= {KAPPA_THRESHOLD} on the overall winner\n")
    print(format_line("overall winner", overall, gate_eligible=True))
    print(format_line("pooled criteria", pooled))
    print()
    for criterion in CRITERIA:
        print(format_line(criterion, per_criterion[criterion]))
    print("\n  Only the overall winner is gate-eligible. The pooled and per-criterion lines show "
          "where\n  each kappa sits against the same threshold, which is supporting detail rather "
          "than a\n  pass or a fail.")
    if unrated:
        print(f"\n  {len(unrated)} pair(s) left entirely unrated: {', '.join(unrated)}")

    print(f"\n  GATE: {'PASS' if gate_pass else 'FAIL'}")
    if gate_pass:
        print("  Layer 3 clears the pre-registered threshold and counts as a primary quality "
              "signal.")
    else:
        print("  Layer 3 does not clear the threshold, so per the proposal it is reported "
              "descriptively rather than as a primary quality signal.")

    result = {
        "generation_run_id": args.run_id,
        "threshold": KAPPA_THRESHOLD,
        "gate_metric": "overall winner",
        "layer3_validated": gate_pass,
        "overall": overall,
        "pooled_criteria": pooled,
        "per_criterion": per_criterion,
        "n_pairs_in_key": len(keys),
        "n_pairs_rated_overall": len(overall_pairs),
        "overall_pairs": overall_pairs,
        "unrated_pairs": unrated,
        "notes": (
            "Researcher answers were translated from sheet positions into side names via key.json "
            "before comparison. Cells were dropped, never imputed, where a rating was blank or the "
            "judge's reconciled winner was null; each kappa reports its own n."
        ),
    }
    write_json(out_path, result)
    print(f"\n  written to {out_path}")
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
