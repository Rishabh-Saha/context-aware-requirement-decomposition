import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_csv import (  # noqa: E402
    cell_metric_rows,
    judgment_rows,
    raw_judgment_rows,
    story_rows,
    plain_summary_rows,
    reported_numbers_rows,
    table_4_3_rows,
    table_4_4_rows,
    table_4_5_rows,
    write_csv,
)


def _diagnostic(issue_key="PIG-1", condition="vanilla", **layer2):
    return {
        "run_id": "RUN", "issue_key": issue_key, "condition": condition,
        "generator_model": "gpt-4o", "generator_temperature": 0.0, "summaries_incomplete": False,
        "layer1": {"num_user_stories": 2, "well_formedness_rate": 1.0, "avg_acceptance_criteria": 3,
                   "file_reference_specificity": 1.0, "inter_story_dependency_rate": 0.5,
                   "dangling_dependency_count": 0},
        "layer2": {"total": 3, "real_rate": 0.3333, "ambiguous_rate": 0.0, "hallucinated_rate": 0.6667,
                   "counts": {"REAL": 1, "AMBIGUOUS": 0, "HALLUCINATED": 2},
                   "unique_references": 2, "tracked_paths_in_window": 733, **layer2},
    }


def test_cell_metrics_prefix_layers_separately():
    """Layer 1 and Layer 2 share a row but never share a column name.

    The methodology reports shape and grounding separately, so an unprefixed `real_rate` sitting
    beside an unprefixed `well_formedness_rate` would be the first step toward averaging them.
    """
    row = cell_metric_rows([_diagnostic()])[0]
    assert row["layer1_num_user_stories"] == 2
    assert row["layer2_hallucinated_rate"] == 0.6667
    assert row["layer2_count_hallucinated"] == 2
    assert not any(k.startswith("real") or k.startswith("well") for k in row)


def test_cell_metrics_sorted_by_issue_then_condition_order():
    rows = cell_metric_rows([
        _diagnostic("PIG-2", "vanilla"),
        _diagnostic("PIG-1", "full_rag"),
        _diagnostic("PIG-1", "vanilla"),
    ])
    assert [(r["issue_key"], r["condition"]) for r in rows] == [
        ("PIG-1", "vanilla"), ("PIG-1", "full_rag"), ("PIG-2", "vanilla"),
    ]


def test_judgment_inconsistent_criterion_stays_empty_not_tie():
    """A position flip leaves the winner blank and flags the criterion, rather than reading as a tie."""
    row = judgment_rows([{
        "run_id": "RUN", "issue_key": "PIG-1", "sub_question": "SQ1",
        "left": "vanilla", "right": "full_rag", "winner": "full_rag",
        "positional_inconsistent": False,
        "per_criterion_winner": {"actionability": None, "completeness": "full_rag"},
        "per_criterion_inconsistent": ["actionability"],
    }])[0]
    assert row["winner_actionability"] is None
    assert row["inconsistent_actionability"] is True
    assert row["winner_completeness"] == "full_rag"
    assert row["inconsistent_completeness"] is False


def test_raw_judgment_keeps_both_orders_and_rationale():
    rows = raw_judgment_rows([
        {"issue_key": "PIG-1", "sub_question": "SQ1", "presentation": "BA", "shown_a": "full_rag",
         "shown_b": "vanilla", "raw_winner": "A", "winner_side": "full_rag",
         "raw_per_criterion": {"clarity": "A"}, "per_criterion_side": {"clarity": "full_rag"},
         "rationale": "second"},
        {"issue_key": "PIG-1", "sub_question": "SQ1", "presentation": "AB", "shown_a": "full_rag",
         "shown_b": "vanilla", "raw_winner": "B", "winner_side": "vanilla",
         "raw_per_criterion": {"clarity": "B"}, "per_criterion_side": {"clarity": "vanilla"},
         "rationale": "first"},
    ])
    assert [r["presentation"] for r in rows] == ["AB", "BA"]
    assert rows[0]["raw_clarity"] == "B" and rows[0]["winner_clarity"] == "vanilla"
    assert rows[0]["rationale"] == "first"


def test_table_4_3_normalises_condition_specific_mean():
    """The per-condition mean arrives under a key named after the condition and lands in one column."""
    rows = table_4_3_rows({"table_4_3_sq1_hallucination_rate": {
        "full_rag_minus_past_tickets": {
            "comparison": "full_rag_minus_past_tickets vs full_rag",
            "full_rag_minus_past_tickets_mean": 0.5558, "full_rag_mean": 0.4958,
            "n": 11, "statistic": 22.5, "p_value": 0.3501, "rank_biserial": 0.3182,
            "adjusted_p": 0.7799, "reject_at_0.05": False,
        }
    }})
    assert rows[0]["condition_mean"] == 0.5558
    assert rows[0]["full_rag_mean"] == 0.4958
    assert rows[0]["reject_at_0.05"] is False


def test_table_4_4_normalises_side_specific_win_counts():
    rows = table_4_4_rows({"table_4_4_layer3_pairwise": {
        "vanilla_vs_full_rag": {"sub_question": "SQ2", "n": 20, "wins_vanilla": 9,
                                "wins_full_rag": 7, "positional_inconsistent": 4,
                                "sign_test_decided": 16, "sign_test_p_value": 0.803619},
    }})
    assert rows[0]["left"] == "vanilla" and rows[0]["right"] == "full_rag"
    assert rows[0]["wins_left"] == 9 and rows[0]["wins_right"] == 7


def test_table_4_5_marks_only_overall_as_the_gate():
    """A per-criterion kappa above the threshold must not read as validating Layer 3."""
    rows = table_4_5_rows({"table_4_5_layer4_calibration": {
        "overall": {"kappa": 0.5649, "n": 19, "ci95": [0.1305, 0.9993], "raw_agreement": 0.8421,
                    "threshold": 0.6, "layer3_validated": False},
        "per_criterion": {"clarity": {"kappa": 0.8148, "n": 15, "ci95": [0.4702, 1.0],
                                      "raw_agreement": 0.9333, "threshold": 0.6,
                                      "layer3_validated": True}},
    }})
    assert [r["scope"] for r in rows] == ["overall", "clarity"]
    assert rows[0]["is_gate_metric"] is True and rows[0]["ci95_low"] == 0.1305
    assert rows[1]["is_gate_metric"] is False


def test_story_rows_read_cells_through_cell_stem(tmp_path):
    (tmp_path / "PIG-1__vanilla.json").write_text(json.dumps({
        "run_id": "RUN", "issue_key": "PIG-1", "condition": "vanilla",
        "decomposition": {"epic_summary": "epic", "user_stories": [
            {"id": "US-1", "story": "As a user, I want x so that y.",
             "acceptance_criteria": ["a", "b"], "complexity": "M",
             "dependencies": [], "source_files": ["src/A.java", "src/B.java"]},
            {"id": "US-2", "story": "As a user, I want z so that w.", "acceptance_criteria": [],
             "complexity": "S", "dependencies": ["US-1"], "source_files": []},
        ]},
    }))
    rows = story_rows(tmp_path, [_diagnostic("PIG-1", "vanilla")])
    assert len(rows) == 2
    assert rows[0]["story_position"] == 1
    assert rows[0]["source_files"] == "src/A.java | src/B.java"
    assert rows[0]["num_source_files"] == 2
    assert rows[1]["dependencies"] == "US-1"
    assert rows[1]["num_acceptance_criteria"] == 0


_AGGREGATES = {
    "table_4_1_structural_metrics": {"vanilla": {
        "num_user_stories": 3.9, "well_formedness_rate": 0.7792, "avg_acceptance_criteria": 2.6408,
        "file_reference_specificity": 1.0, "inter_story_dependency_rate": 0.4583,
        "dangling_dependency_count": 0, "n": 20}},
    "table_4_2_layer2_verdicts": {
        "verdict_distribution": {"vanilla": {"real_rate": 0.0667, "ambiguous_rate": 0.23,
                                             "hallucinated_rate": 0.7033, "n": 20}},
        "sq2_hallucination_rate_test": {"comparison": "vanilla vs full_rag", "vanilla_mean": 0.7033,
                                        "full_rag_mean": 0.4958, "n": 17, "statistic": 33.0,
                                        "p_value": 0.0388, "rank_biserial": 0.5686},
    },
    "table_4_4_layer3_pairwise": {"vanilla_vs_full_rag": {
        "sub_question": "SQ2", "n": 20, "wins_vanilla": 9, "wins_full_rag": 7,
        "positional_inconsistent": 4, "sign_test_decided": 16, "sign_test_p_value": 0.803619}},
    "table_4_5_layer4_calibration": {"overall": {
        "kappa": 0.5649, "n": 19, "ci95": [0.1305, 0.9993], "raw_agreement": 0.8421,
        "threshold": 0.6, "layer3_validated": False}},
    "positional_inconsistency_4_6": {"inconsistent": 21, "total": 140, "rate": 0.15},
}


def test_summary_carries_the_layer3_descriptive_warning():
    """A file sent on its own has to say that Layer 3 did not pass its gate.

    Without the note, a reader sees 9 wins against 7 and reads a quality result where the design
    says there is none.
    """
    rows = reported_numbers_rows(_AGGREGATES)
    p_row = next(r for r in rows if r["metric"] == "sign_test_p_value")
    assert "DESCRIPTIVE ONLY" in p_row["note"]
    kappa_row = next(r for r in rows if r["metric"] == "kappa")
    assert "below the 0.60 threshold" in kappa_row["note"]


def test_summary_labels_win_counts_by_side_not_by_position():
    """`wins_left` is meaningless once the row is separated from its table header."""
    metrics = {r["metric"] for r in reported_numbers_rows(_AGGREGATES)}
    assert {"wins_vanilla", "wins_full_rag"} <= metrics
    assert "wins_left" not in metrics


def test_summary_includes_run_configuration_when_the_manifest_is_present():
    manifest = {"run_id": "RUN", "generator": {"model": "gpt-4o", "temperature": 0.0},
                "judge": {"model": "claude-sonnet-4-6"}, "embedding": {"model": "emb"},
                "per_type": 2, "n_requirements": 20, "n_cells": 120, "code_commit": "abc123"}
    rows = reported_numbers_rows(_AGGREGATES, manifest)
    config = {r["metric"]: r["value"] for r in rows if r["section"] == "Run configuration"}
    assert config["model"] in ("gpt-4o", "claude-sonnet-4-6", "emb")  # three rows share the metric name
    assert config["temperature"] == 0.0
    assert config["n_cells"] == 120
    assert reported_numbers_rows(_AGGREGATES) == [r for r in rows if r["section"] != "Run configuration"]


def test_summary_uses_one_shape_for_every_section():
    rows = reported_numbers_rows(_AGGREGATES, {"run_id": "RUN"})
    assert all(list(r.keys()) == ["section", "unit", "metric", "value", "n", "note"] for r in rows)


def test_plain_summary_takes_its_numbers_from_the_aggregates():
    """The sentences are fixed, the numbers inside them are not.

    A hand-written summary drifts the moment a figure is recomputed. Feeding the rates in from
    aggregates.json is what stops the prose and the results disagreeing.
    """
    aggregates = json.loads(json.dumps(_AGGREGATES))
    aggregates["table_4_2_layer2_verdicts"]["verdict_distribution"]["full_rag"] = {
        "real_rate": 0.0583, "ambiguous_rate": 0.4458, "hallucinated_rate": 0.4958, "n": 20,
    }
    rows = plain_summary_rows(aggregates, {"n_requirements": 20, "n_cells": 120})
    grounding = next(r for r in rows if "Invented file references" in r["what was measured"])
    assert "70.3%" in grounding["result"] and "49.6%" in grounding["result"]
    assert "0.0388" in grounding["result"]

    aggregates["table_4_2_layer2_verdicts"]["verdict_distribution"]["vanilla"]["hallucinated_rate"] = 0.5
    changed = plain_summary_rows(aggregates, {})
    assert "50.0%" in next(r for r in changed if "Invented" in r["what was measured"])["result"]


def test_plain_summary_states_the_three_things_that_get_over_read():
    """The failed gate, the order flips, and the single-project scope are rows, not footnotes."""
    text = " ".join(
        r["what it means"] for r in plain_summary_rows(_AGGREGATES, {"n_requirements": 20})
    )
    assert "descriptive only" in text
    assert "presentation order" in text
    assert "case study" in text
    caveats = [r for r in plain_summary_rows(_AGGREGATES, {}) if r["part"] == "Caveat"]
    assert len(caveats) == 3


def test_plain_summary_is_short_and_numbered_in_reading_order():
    rows = plain_summary_rows(_AGGREGATES, {"n_requirements": 20, "n_cells": 120})
    assert len(rows) <= 15, "a summary that needs scrolling is not a summary"
    assert [r["#"] for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["part"] == "Headline"
    assert rows[-1]["part"] == "Caveat"


def test_plain_summary_survives_a_missing_section():
    """A partial aggregates file yields 'n/a' rather than a traceback midway through the export."""
    rows = plain_summary_rows({}, {})
    assert any("n/a" in r["result"] for r in rows)
    assert len(rows) == 13


def test_write_csv_roundtrips_through_the_stdlib_reader(tmp_path):
    """A story containing a comma and a quote has to survive the round trip."""
    rows = [{"a": 'has, comma and "quotes"', "b": 1}]
    path = write_csv(tmp_path / "out" / "x.csv", rows, ["a", "b"])
    back = list(csv.DictReader(open(path, encoding="utf-8")))
    assert back == [{"a": 'has, comma and "quotes"', "b": "1"}]
