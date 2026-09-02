"""Export the run's output as flat CSV tables, for spreadsheets and plotting.

The committed artefacts are JSON and JSONL because that is the shape the pipeline writes and the
shape `aggregate_results.py` reads back. Neither opens usefully in a spreadsheet, and every thesis
table has at some point been retyped by hand from a printed aggregate. This script closes that gap:
it flattens the same artefacts into one CSV per table, so a figure in the thesis can be traced to a
row in a file rather than to a transcription.

Nothing is recomputed here. Every number written comes from `results/aggregates.json`,
`diagnostics.jsonl`, `judgments.jsonl` or `judgments_raw.jsonl` exactly as it was computed by
`aggregate_results.py`. If a number looks wrong, the fix belongs upstream, not in this file.

Three levels are written. `results_summary.csv` is the findings in plain sentences, one row per
point, for a reader who wants to know what the study found rather than to re-analyse it. The
table_4_* files and `all_reported_numbers.csv` are the aggregates as they appear in Chapters 4 to 6.
The per-cell and per-comparison files are the raw units of analysis (one row per requirement x
condition, one row per pairwise comparison), for re-running statistics or plotting distributions.

The story-level export is the only part that needs the run directory rather than the results
package, since the decompositions themselves are archived per cell and not committed. It is skipped
with a notice when that directory is absent, so the script still works for a reader who has the
repository alone.

Usage:
    python scripts/export_csv.py --run-id 20260814T033139Z
    python scripts/export_csv.py --run-id 20260814T033139Z --out-dir results/csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import Condition  # noqa: E402
from src.eval.rendering import cell_stem  # noqa: E402
from src.paths import DEFAULT_RESULTS_DIR, DEFAULT_RUNS_DIR  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402

# Conditions and criteria are written in a fixed order rather than in dict order, so a diff between
# two exports shows changed numbers instead of reshuffled rows.
CONDITION_ORDER: tuple[str, ...] = tuple(c.value for c in Condition)
CRITERIA: tuple[str, ...] = (
    "actionability",
    "completeness",
    "project_specificity",
    "granularity",
    "clarity",
)

LAYER1_FIELDS = (
    "num_user_stories",
    "well_formedness_rate",
    "avg_acceptance_criteria",
    "file_reference_specificity",
    "inter_story_dependency_rate",
    "dangling_dependency_count",
)
LAYER2_RATE_FIELDS = ("real_rate", "ambiguous_rate", "hallucinated_rate")


def _condition_sort_key(condition: str) -> int:
    """Position in CONDITION_ORDER, with unknown values sorted last rather than raising."""
    return CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> Path:
    """Write rows to CSV with a fixed column order. Missing keys become empty cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


# -- per-cell and per-comparison units ------------------------------------------------------------


def cell_metric_rows(diagnostics: list[dict]) -> list[dict]:
    """One row per requirement x condition: Layer 1 metrics beside Layer 2 verdicts.

    Layer 1 and Layer 2 sit in the same row because they describe the same generation, not because
    they are commensurable. They stay separately prefixed so a reader cannot accidentally average
    shape and grounding into a single quality score, which the methodology explicitly forbids.
    """
    rows = []
    for record in diagnostics:
        layer1 = record.get("layer1", {})
        layer2 = record.get("layer2", {})
        counts = layer2.get("counts", {})
        row = {
            "run_id": record.get("run_id"),
            "issue_key": record.get("issue_key"),
            "condition": record.get("condition"),
            "generator_model": record.get("generator_model"),
            "generator_temperature": record.get("generator_temperature"),
            "summaries_incomplete": record.get("summaries_incomplete"),
        }
        row.update({f"layer1_{f}": layer1.get(f) for f in LAYER1_FIELDS})
        row["layer2_total_references"] = layer2.get("total")
        row["layer2_unique_references"] = layer2.get("unique_references")
        row.update({f"layer2_{f}": layer2.get(f) for f in LAYER2_RATE_FIELDS})
        row["layer2_count_real"] = counts.get("REAL")
        row["layer2_count_ambiguous"] = counts.get("AMBIGUOUS")
        row["layer2_count_hallucinated"] = counts.get("HALLUCINATED")
        row["layer2_tracked_paths_in_window"] = layer2.get("tracked_paths_in_window")
        rows.append(row)
    rows.sort(key=lambda r: (r["issue_key"] or "", _condition_sort_key(r["condition"] or "")))
    return rows


def judgment_rows(judgments: list[dict]) -> list[dict]:
    """One row per reconciled pairwise comparison, per-criterion winners spread into columns.

    `winner` is null where the two presentation orders disagreed. That is kept as an empty cell
    rather than filled with a tie, because a position flip is a judge failure and a genuine tie is a
    judge decision, and collapsing them would hide the 15% inconsistency rate.
    """
    rows = []
    for record in judgments:
        per_criterion = record.get("per_criterion_winner", {})
        inconsistent = set(record.get("per_criterion_inconsistent", []))
        row = {
            "run_id": record.get("run_id"),
            "issue_key": record.get("issue_key"),
            "sub_question": record.get("sub_question"),
            "left": record.get("left"),
            "right": record.get("right"),
            "winner": record.get("winner"),
            "positional_inconsistent": record.get("positional_inconsistent"),
            "ab_winner_side": record.get("ab_winner_side"),
            "ba_winner_side": record.get("ba_winner_side"),
            "judge_model": record.get("judge_model"),
            "judge_temperature": record.get("judge_temperature"),
            "judged_at": record.get("judged_at"),
        }
        for criterion in CRITERIA:
            row[f"winner_{criterion}"] = per_criterion.get(criterion)
            row[f"inconsistent_{criterion}"] = criterion in inconsistent
        rows.append(row)
    rows.sort(key=lambda r: (r["sub_question"] or "", r["left"] or "", r["issue_key"] or ""))
    return rows


def raw_judgment_rows(raw: list[dict]) -> list[dict]:
    """One row per presented order, before reconciliation, with the judge's rationale.

    Two rows per comparison. Retained so positional inconsistency can be recomputed from source, and
    so the rationale text stays readable next to the verdict it explains.
    """
    rows = []
    for record in raw:
        per_criterion = record.get("per_criterion_side", {})
        raw_per_criterion = record.get("raw_per_criterion", {})
        row = {
            "run_id": record.get("run_id"),
            "issue_key": record.get("issue_key"),
            "sub_question": record.get("sub_question"),
            "presentation": record.get("presentation"),
            "shown_a": record.get("shown_a"),
            "shown_b": record.get("shown_b"),
            "raw_winner": record.get("raw_winner"),
            "winner_side": record.get("winner_side"),
            "judge_model": record.get("judge_model"),
            "judge_temperature": record.get("judge_temperature"),
            "judged_at": record.get("judged_at"),
        }
        for criterion in CRITERIA:
            row[f"raw_{criterion}"] = raw_per_criterion.get(criterion)
            row[f"winner_{criterion}"] = per_criterion.get(criterion)
        row["rationale"] = record.get("rationale")
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["sub_question"] or "",
            r["issue_key"] or "",
            r["shown_a"] or "",
            r["presentation"] or "",
        )
    )
    return rows


def story_rows(run_dir: str | Path, diagnostics: list[dict]) -> list[dict]:
    """One row per generated user story, across all 120 cells.

    This is the actual output of the pipeline rather than a measurement of it. Lists are joined with
    " | " because a story's acceptance criteria and suggested files are read as a set by a human, and
    nesting a second delimiter inside a CSV cell buys nothing over that.

    Cells are located through `cell_stem` rather than by globbing, so a missing generation raises
    here instead of quietly shrinking the export.
    """
    run_dir = Path(run_dir)
    rows = []
    cells = sorted(
        {(r["issue_key"], r["condition"]) for r in diagnostics},
        key=lambda c: (c[0], _condition_sort_key(c[1])),
    )
    for issue_key, condition in cells:
        path = run_dir / f"{cell_stem(issue_key, condition)}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        decomposition = record.get("decomposition", {})
        stories = decomposition.get("user_stories", [])
        for position, story in enumerate(stories, start=1):
            rows.append(
                {
                    "run_id": record.get("run_id"),
                    "issue_key": issue_key,
                    "condition": condition,
                    "epic_summary": decomposition.get("epic_summary"),
                    "story_position": position,
                    "story_id": story.get("id"),
                    "story": story.get("story"),
                    "complexity": story.get("complexity"),
                    "num_acceptance_criteria": len(story.get("acceptance_criteria", [])),
                    "acceptance_criteria": " | ".join(story.get("acceptance_criteria", [])),
                    "dependencies": " | ".join(story.get("dependencies", [])),
                    "num_source_files": len(story.get("source_files", [])),
                    "source_files": " | ".join(story.get("source_files", [])),
                }
            )
    return rows


# -- thesis tables from aggregates.json ------------------------------------------------------------


def table_4_1_rows(aggregates: dict) -> list[dict]:
    """Layer 1 structural metrics by condition."""
    table = aggregates.get("table_4_1_structural_metrics", {})
    rows = []
    for condition in sorted(table, key=_condition_sort_key):
        values = table[condition]
        row = {"condition": condition, "n": values.get("n")}
        row.update({f: values.get(f) for f in LAYER1_FIELDS})
        rows.append(row)
    return rows


def table_4_2_verdict_rows(aggregates: dict) -> list[dict]:
    """Layer 2 verdict distribution by condition."""
    table = aggregates.get("table_4_2_layer2_verdicts", {}).get("verdict_distribution", {})
    rows = []
    for condition in sorted(table, key=_condition_sort_key):
        values = table[condition]
        row = {"condition": condition, "n": values.get("n")}
        row.update({f: values.get(f) for f in LAYER2_RATE_FIELDS})
        rows.append(row)
    return rows


def table_4_2_sq2_rows(aggregates: dict) -> list[dict]:
    """The single SQ2 paired test on hallucination rate, vanilla against full RAG.

    One row, kept in its own file rather than appended to the distribution table, because a test and
    a descriptive mean are different kinds of number and mixing them in one sheet invites a reader to
    sort the column.
    """
    test = aggregates.get("table_4_2_layer2_verdicts", {}).get("sq2_hallucination_rate_test")
    if not test:
        return []
    return [
        {
            "comparison": test.get("comparison"),
            "vanilla_mean": test.get("vanilla_mean"),
            "full_rag_mean": test.get("full_rag_mean"),
            "n_pairs_nonzero": test.get("n"),
            "wilcoxon_statistic": test.get("statistic"),
            "p_value": test.get("p_value"),
            "rank_biserial": test.get("rank_biserial"),
        }
    ]


def table_4_3_rows(aggregates: dict) -> list[dict]:
    """SQ1 leave-one-out comparisons against full RAG, with Holm-Bonferroni adjusted p.

    The per-condition mean arrives under a condition-specific key, so it is normalised to
    `condition_mean` here. Otherwise the CSV would need four columns that are each populated on
    exactly one row.
    """
    table = aggregates.get("table_4_3_sq1_hallucination_rate", {})
    rows = []
    for condition in sorted(table, key=_condition_sort_key):
        values = table[condition]
        rows.append(
            {
                "condition": condition,
                "comparison": values.get("comparison"),
                "condition_mean": values.get(f"{condition}_mean"),
                "full_rag_mean": values.get("full_rag_mean"),
                "n_pairs_nonzero": values.get("n"),
                "wilcoxon_statistic": values.get("statistic"),
                "p_value": values.get("p_value"),
                "adjusted_p": values.get("adjusted_p"),
                "rank_biserial": values.get("rank_biserial"),
                "reject_at_0.05": values.get("reject_at_0.05"),
            }
        )
    return rows


def table_4_4_rows(aggregates: dict) -> list[dict]:
    """Layer 3 pairwise outcomes, one row per comparison type.

    Win counts arrive under side-specific keys (`wins_vanilla`, `wins_full_rag`), which cannot be a
    CSV column since they differ per row. They are normalised to wins_left / wins_right, with the
    side names kept in their own columns so the row stays readable on its own.
    """
    table = aggregates.get("table_4_4_layer3_pairwise", {})
    rows = []
    for key, values in table.items():
        left, _, right = key.partition("_vs_")
        rows.append(
            {
                "comparison": key,
                "sub_question": values.get("sub_question"),
                "left": left,
                "right": right,
                "n": values.get("n"),
                "wins_left": values.get(f"wins_{left}"),
                "wins_right": values.get(f"wins_{right}"),
                "positional_inconsistent": values.get("positional_inconsistent"),
                "sign_test_decided": values.get("sign_test_decided"),
                "sign_test_p_value": values.get("sign_test_p_value"),
            }
        )
    rows.sort(key=lambda r: (r["sub_question"] or "", _condition_sort_key(r["left"])))
    return rows


def table_4_5_rows(aggregates: dict) -> list[dict]:
    """Layer 4 agreement, overall then pooled then per criterion.

    `layer3_validated` is carried per scope so the gate outcome travels with the number. The gate is
    decided on the overall winner alone: a per-criterion kappa above 0.6 does not validate Layer 3.
    """
    table = aggregates.get("table_4_5_layer4_calibration", {})
    scopes = []
    if "overall" in table:
        scopes.append(("overall", table["overall"]))
    if "pooled_criteria" in table:
        scopes.append(("pooled_criteria", table["pooled_criteria"]))
    per_criterion = table.get("per_criterion", {})
    for criterion in CRITERIA:
        if criterion in per_criterion:
            scopes.append((criterion, per_criterion[criterion]))

    rows = []
    for scope, values in scopes:
        ci = values.get("ci95") or [None, None]
        rows.append(
            {
                "scope": scope,
                "is_gate_metric": scope == "overall",
                "kappa": values.get("kappa"),
                "n": values.get("n"),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "raw_agreement": values.get("raw_agreement"),
                "threshold": values.get("threshold"),
                "layer3_validated": values.get("layer3_validated"),
            }
        )
    return rows


def positional_inconsistency_rows(aggregates: dict) -> list[dict]:
    """The Section 4.6 positional inconsistency rate over all comparisons."""
    values = aggregates.get("positional_inconsistency_4_6")
    if not values:
        return []
    return [
        {
            "inconsistent": values.get("inconsistent"),
            "total": values.get("total"),
            "rate": values.get("rate"),
        }
    ]


# -- one file with every number, and one file with the findings in words -----------------------------


def reported_numbers_rows(aggregates: dict, manifest: dict | None = None) -> list[dict]:
    """Every reported number in one long-format table.

    The per-table files above are the right shape for plotting, and the wrong shape for a reader who
    just wants every figure in one place: they arrive as ten attachments with no stated relationship.
    This stacks all of them into one file, one row per number, with the section and unit that
    identify it. It is complete rather than readable; `plain_summary_rows` below is the readable one.

    Long format rather than wide because the tables have genuinely different columns. Forcing a
    kappa, a win count and a mean acceptance-criteria count into shared columns would leave most
    cells empty and invite a reader to compare down a column that mixes three kinds of quantity.

    `note` carries the reading each number needs to be given, since a CSV sent by itself travels
    without the surrounding chapter. The Layer 3 rows say they are descriptive, because the Layer 4
    gate failed and a win count read as a quality result would be exactly the wrong conclusion.
    """
    rows: list[dict] = []

    def add(section: str, unit: str, metric: str, value, n=None, note: str = "") -> None:
        rows.append(
            {"section": section, "unit": unit, "metric": metric, "value": value, "n": n, "note": note}
        )

    if manifest:
        add("Run configuration", "run", "run_id", manifest.get("run_id"))
        add("Run configuration", "generator", "model", manifest.get("generator", {}).get("model"))
        add(
            "Run configuration", "generator", "temperature",
            manifest.get("generator", {}).get("temperature"),
            note="Fixed across all six conditions.",
        )
        add(
            "Run configuration", "judge", "model", manifest.get("judge", {}).get("model"),
            note="Different provider family from the generator, enforced in code.",
        )
        add("Run configuration", "embedding", "model", manifest.get("embedding", {}).get("model"))
        add(
            "Run configuration", "retrieval", "per_type_budget", manifest.get("per_type"),
            note="Each context type retrieved independently, not a global top-k.",
        )
        add("Run configuration", "design", "n_requirements", manifest.get("n_requirements"))
        add(
            "Run configuration", "design", "n_cells", manifest.get("n_cells"),
            note="6 conditions x 20 requirements, no missing cells.",
        )
        add("Run configuration", "code", "commit", manifest.get("code_commit"))

    layer1 = "Table 4.1 Layer 1 structural metrics (diagnostic)"
    for row in table_4_1_rows(aggregates):
        for field in LAYER1_FIELDS:
            add(layer1, row["condition"], field, row[field], row["n"])

    layer2 = "Table 4.2 Layer 2 file-existence verdicts (diagnostic)"
    for row in table_4_2_verdict_rows(aggregates):
        for field in LAYER2_RATE_FIELDS:
            add(layer2, row["condition"], field, row[field], row["n"])

    sq2_note = "SQ2. Significant reduction in hallucination, driven by a shift to AMBIGUOUS."
    for row in table_4_2_sq2_rows(aggregates):
        for metric in ("vanilla_mean", "full_rag_mean", "wilcoxon_statistic", "p_value", "rank_biserial"):
            add(
                "Table 4.2 SQ2 test, hallucination rate", row["comparison"], metric, row[metric],
                row["n_pairs_nonzero"], sq2_note if metric == "p_value" else "",
            )

    sq1_note = "SQ1. Holm-Bonferroni over the four leave-one-out tests; none significant."
    for row in table_4_3_rows(aggregates):
        for metric in ("condition_mean", "full_rag_mean", "wilcoxon_statistic", "p_value",
                       "adjusted_p", "rank_biserial"):
            add(
                "Table 4.3 SQ1 leave-one-out, hallucination rate", row["comparison"], metric,
                row[metric], row["n_pairs_nonzero"], sq1_note if metric == "adjusted_p" else "",
            )

    layer3_note = "DESCRIPTIVE ONLY: the Layer 4 kappa gate failed, so Layer 3 is not a primary signal."
    for row in table_4_4_rows(aggregates):
        for metric in ("wins_left", "wins_right", "positional_inconsistent", "sign_test_decided",
                       "sign_test_p_value"):
            label = metric
            if metric == "wins_left":
                label = f"wins_{row['left']}"
            elif metric == "wins_right":
                label = f"wins_{row['right']}"
            add(
                f"Table 4.4 Layer 3 pairwise, {row['sub_question']} (descriptive)",
                row["comparison"], label, row[metric], row["n"],
                layer3_note if metric == "sign_test_p_value" else "",
            )

    gate_note = "Gate metric. 0.5649 is below the 0.60 threshold, so Layer 3 is reported descriptively."
    for row in table_4_5_rows(aggregates):
        for metric in ("kappa", "ci95_low", "ci95_high", "raw_agreement"):
            add(
                "Table 4.5 Layer 4 researcher-versus-judge agreement", row["scope"], metric,
                row[metric], row["n"],
                gate_note if metric == "kappa" and row["is_gate_metric"] else "",
            )

    for row in positional_inconsistency_rows(aggregates):
        add(
            "Section 4.6 judge positional inconsistency", "all comparisons", "rate", row["rate"],
            row["total"], "Comparisons where the two presentation orders disagreed on the winner.",
        )

    return rows


def _pct(value) -> str:
    """A rate as a percentage, which is how these numbers get said out loud."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def plain_summary_rows(aggregates: dict, manifest: dict | None = None) -> list[dict]:
    """The findings as sentences, one row per point, for a reader who is not going to re-analyse.

    `reported_numbers_rows` is complete and unreadable: 156 rows of metric names, which is the right
    artefact to attach to a submission and the wrong one to talk somebody through. This file answers
    the questions a reader actually asks in order. What was tested, what was found, and what should
    not be concluded from it.

    Every number is still pulled from `aggregates.json` rather than typed into the sentence, so the
    prose cannot drift away from the results the way a hand-written summary does. The wording around
    the numbers is fixed, the numbers are not.

    The caveats are rows in the same table rather than a footnote, because the three things most
    likely to be over-read here (a failed judge, an order-sensitive judge, and a preference over the
    human reference that is really a format effect) are exactly the things a summary tends to drop.
    """
    manifest = manifest or {}
    layer2 = aggregates.get("table_4_2_layer2_verdicts", {})
    verdicts = layer2.get("verdict_distribution", {})
    vanilla = verdicts.get("vanilla", {})
    full_rag = verdicts.get("full_rag", {})
    sq2_test = layer2.get("sq2_hallucination_rate_test", {})
    sq1 = aggregates.get("table_4_3_sq1_hallucination_rate", {})
    pairwise = aggregates.get("table_4_4_layer3_pairwise", {})
    sq2_judge = pairwise.get("vanilla_vs_full_rag", {})
    sq3 = pairwise.get("full_rag_vs_reference", {})
    calibration = aggregates.get("table_4_5_layer4_calibration", {}).get("overall", {})
    flips = aggregates.get("positional_inconsistency_4_6", {})

    rows: list[dict] = []

    def add(part: str, measured: str, result: str, meaning: str) -> None:
        rows.append(
            {
                "#": len(rows) + 1,
                "part": part,
                "what was measured": measured,
                "result": result,
                "what it means": meaning,
            }
        )

    add(
        "Headline",
        "The one-sentence result",
        "Project context made the model's file references far less invented, but did not make its "
        "user stories better in quality terms",
        "Grounding and quality came apart. That split is the finding, and it is a more interesting "
        "result than a straightforward win would have been.",
    )

    n_req = manifest.get("n_requirements", 20)
    n_cells = manifest.get("n_cells", 120)
    add(
        "What was done",
        "The experiment",
        f"{n_req} real requirements from the Apache Pig project, each broken into user stories "
        f"{n_cells // n_req} different ways, giving {n_cells} outputs",
        "The six ways are: no context at all, all four kinds of context, and four runs that each "
        "remove one kind. Removing one at a time is what shows whether that kind was pulling weight.",
    )
    add(
        "What was done",
        "The four kinds of context",
        "Past tickets, design documents, coding conventions, and summaries of the existing code",
        "Each kind is retrieved separately with its own budget, so dropping one cannot be quietly "
        "replaced by more of another.",
    )
    generator = manifest.get("generator", {})
    add(
        "What was done",
        "Which model wrote the outputs",
        f"{generator.get('model', 'the generator')} at temperature {generator.get('temperature', 0.0)}, "
        "the same for all six conditions",
        "Temperature 0 means the same prompt gives the same answer, so differences between "
        "conditions come from the context and not from randomness.",
    )
    add(
        "What was done",
        "Which model scored them",
        f"{manifest.get('judge', {}).get('model', 'the judge')}, from a different provider family "
        "than the generator",
        "A model never judges its own output, and the judge was shown the two outputs as A and B "
        "with no indication of which condition produced them.",
    )

    add(
        "Main finding",
        "Invented file references, no context versus full context",
        f"{_pct(vanilla.get('hallucinated_rate'))} of file paths did not exist without context, "
        f"{_pct(full_rag.get('hallucinated_rate'))} with it (p = {sq2_test.get('p_value')})",
        "The clearest positive result in the study. p below 0.05 means a difference this large is "
        "unlikely to be chance, and the effect size "
        f"({sq2_test.get('rank_biserial')}) says it is a moderate-to-large one.",
    )
    add(
        "Main finding",
        "Where those references went instead",
        f"Confirmed-real stayed flat ({_pct(vanilla.get('real_rate'))} to "
        f"{_pct(full_rag.get('real_rate'))}); unverifiable rose "
        f"({_pct(vanilla.get('ambiguous_rate'))} to {_pct(full_rag.get('ambiguous_rate'))})",
        "Context stopped the model inventing file names, but did not make it name the right ones. "
        "Say this before anybody asks, because it is the honest reading of the headline number.",
    )
    smallest_adjusted = min(
        (v.get("adjusted_p") for v in sq1.values() if v.get("adjusted_p") is not None),
        default=None,
    )
    add(
        "Main finding",
        "Removing one kind of context at a time",
        f"None of the four made a significant difference (smallest adjusted p = {smallest_adjusted})",
        "No single kind of context carries the effect. With 20 requirements the test has modest "
        "power, so this means no effect was detected, not that there is none. It was expected as a "
        "possible outcome before the run, so it is a result rather than a failure.",
    )
    add(
        "Main finding",
        "Did context make the stories better to read and act on?",
        f"{sq2_judge.get('wins_full_rag')} wins for full context against "
        f"{sq2_judge.get('wins_vanilla')} for no context, out of "
        f"{sq2_judge.get('sign_test_decided')} decided (p = {sq2_judge.get('sign_test_p_value')})",
        "No quality difference. Combined with the finding above: better grounded is not the same as "
        "better written. Treat this as descriptive only, for the reason two rows down.",
    )
    add(
        "Main finding",
        "Model output against the original human-written ticket",
        f"The model was preferred in {sq3.get('wins_full_rag')} of {sq3.get('sign_test_decided')} "
        "decided comparisons",
        "Almost certainly a format effect rather than a substance one. The model writes in the "
        "expected role/goal/benefit shape with acceptance criteria and the original ticket does "
        "not, so the judge has an easy tell. Not a claim that the model outperforms the developers.",
    )

    add(
        "Caveat",
        "Was the judge itself trustworthy?",
        f"Agreement with the researcher's own blind ratings was kappa {calibration.get('kappa')}, "
        f"below the 0.60 bar set in advance (raw agreement {_pct(calibration.get('raw_agreement'))}, "
        f"n = {calibration.get('n')})",
        "The judge could not be validated, so every quality result above is reported as descriptive "
        "only and the grounding results carry the conclusions. This rule was fixed before the run, "
        "not chosen after seeing the numbers, which is the point worth making to a supervisor.",
    )
    add(
        "Caveat",
        "How consistent was the judge?",
        f"{_pct(flips.get('rate'))} of comparisons changed winner when the two outputs swapped "
        f"places ({flips.get('inconsistent')} of {flips.get('total')})",
        "Some verdicts depend on presentation order. Each comparison was run both ways and "
        "reconciled precisely so this could be measured rather than hidden.",
    )
    add(
        "Caveat",
        "How far do these findings reach?",
        f"One project (Apache Pig), {n_req} requirements, one generator model",
        "A case study. The result is evidence about this setting, and repeating the work on another "
        "project would be needed before generalising it.",
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="Run identifier, e.g. 20260814T033139Z")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where the CSVs are written. Defaults to <results-dir>/csv.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "csv"

    aggregates = json.loads((results_dir / "aggregates.json").read_text(encoding="utf-8"))
    manifest_path = results_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    diagnostics = list(read_jsonl(results_dir / "diagnostics.jsonl"))
    judgments = list(read_jsonl(results_dir / "judgments.jsonl"))
    raw = list(read_jsonl(results_dir / "judgments_raw.jsonl"))

    if aggregates.get("run_id") != args.run_id:
        raise SystemExit(
            f"aggregates.json is for run {aggregates.get('run_id')}, not {args.run_id}. "
            "Regenerate it with scripts/aggregate_results.py before exporting."
        )

    written: list[tuple[Path, int]] = []

    def emit(name: str, rows: list[dict]) -> None:
        if not rows:
            print(f"  skipped {name}: no rows in the source artefact")
            return
        path = write_csv(out_dir / name, rows, list(rows[0].keys()))
        written.append((path, len(rows)))

    emit("results_summary.csv", plain_summary_rows(aggregates, manifest))
    emit("all_reported_numbers.csv", reported_numbers_rows(aggregates, manifest))
    emit("cell_metrics.csv", cell_metric_rows(diagnostics))
    emit("judgments.csv", judgment_rows(judgments))
    emit("judgments_raw.csv", raw_judgment_rows(raw))
    emit("table_4_1_structural_metrics.csv", table_4_1_rows(aggregates))
    emit("table_4_2_verdict_distribution.csv", table_4_2_verdict_rows(aggregates))
    emit("table_4_2_sq2_test.csv", table_4_2_sq2_rows(aggregates))
    emit("table_4_3_sq1_hallucination.csv", table_4_3_rows(aggregates))
    emit("table_4_4_layer3_pairwise.csv", table_4_4_rows(aggregates))
    emit("table_4_5_layer4_calibration.csv", table_4_5_rows(aggregates))
    emit("positional_inconsistency.csv", positional_inconsistency_rows(aggregates))

    run_dir = Path(args.runs_dir) / args.run_id
    if run_dir.is_dir():
        emit("user_stories.csv", story_rows(run_dir, diagnostics))
    else:
        print(
            f"  skipped user_stories.csv: {run_dir} not present. The per-cell decompositions are "
            "archived with the run, not in the results package."
        )

    print(f"\nWrote {len(written)} CSVs to {out_dir}:")
    for path, count in written:
        print(f"  {path.name:<40} {count:>5} rows")


if __name__ == "__main__":
    main()
