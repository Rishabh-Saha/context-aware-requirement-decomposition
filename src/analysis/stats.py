"""Statistical procedures for the analysis plan (proposal Section 7.5).

Everything the plan pre-specifies lives here so the analysis notebooks stay thin and the tests
pin the behaviour:

    paired Wilcoxon signed-rank      per-metric comparison of conditions (SQ1, SQ2)
    rank-biserial correlation        effect size reported alongside every Wilcoxon test
    Holm-Bonferroni step-down        family-wise error control across the four SQ1 pair tests
    Cohen's kappa                    Layer 4 judge-vs-researcher agreement (95% CI)

Kept dependency-light: SciPy and scikit-learn are imported lazily so the pure helpers here run
even in a minimal environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    effect_size: float          # rank-biserial correlation
    n: int


def rank_biserial_from_pairs(x: list[float], y: list[float]) -> float:
    """Matched-pairs rank-biserial correlation: (favourable - unfavourable) / total signed ranks.

    Positive means x tends to exceed y. Ties (zero differences) are dropped, matching the default
    Wilcoxon handling.
    """
    diffs = [a - b for a, b in zip(x, y) if (a - b) != 0]
    n = len(diffs)
    if n == 0:
        return 0.0
    ranks = _rankdata([abs(d) for d in diffs])
    pos = sum(r for d, r in zip(diffs, ranks) if d > 0)
    neg = sum(r for d, r in zip(diffs, ranks) if d < 0)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> WilcoxonResult:
    if len(x) != len(y):
        raise ValueError("x and y must be the same length (paired)")
    from scipy.stats import wilcoxon  # lazy

    non_tied = [(a, b) for a, b in zip(x, y) if a != b]
    n = len(non_tied)
    if n == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, effect_size=0.0, n=0)
    stat, p = wilcoxon(x, y, zero_method="wilcox")
    eff = rank_biserial_from_pairs(x, y)
    return WilcoxonResult(statistic=float(stat), p_value=float(p), effect_size=eff, n=n)


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down. Returns one record per input p-value (original order) with its
    adjusted p-value and reject decision, controlling the family-wise error rate."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, adj)   # enforce monotonic non-decreasing adjusted p
        adjusted[idx] = running_max
    return [
        {"index": i, "p_value": pvalues[i], "adjusted_p": adjusted[i], "reject": adjusted[i] < alpha}
        for i in range(m)
    ]


def cohens_kappa(rater_a: list, rater_b: list, ci: bool = True) -> dict:
    """Cohen's kappa with a 95% CI from the standard large-sample variance.
 
    rater_a / rater_b are parallel label sequences. Labels may be any hashable value; the
    agreement table is built over the union of labels actually observed.
    """
    from sklearn.metrics import cohen_kappa_score  # lazy
 
    n = len(rater_a)
    k = float(cohen_kappa_score(rater_a, rater_b))
    result = {"kappa": round(k, 4), "n": n}
    if not ci or n <= 1:
        return result
 
    labels = sorted({*rater_a, *rater_b}, key=str)
    idx = {lab: i for i, lab in enumerate(labels)}
    m = len(labels)
 
    # Observed proportions p[i][j], row marginals p_i., column marginals p_.j
    p = [[0.0] * m for _ in range(m)]
    for a, b in zip(rater_a, rater_b):
        p[idx[a]][idx[b]] += 1.0 / n
    row = [sum(p[i]) for i in range(m)]
    col = [sum(p[i][j] for i in range(m)) for j in range(m)]
 
    po = sum(p[i][i] for i in range(m))
    pe = sum(row[i] * col[i] for i in range(m))
    if abs(1.0 - pe) < 1e-12:
        return result
 
    # Standard asymptotic variance of kappa.
    term1 = sum(
        p[i][i] * ((1.0 - pe) - (row[i] + col[i]) * (1.0 - po)) ** 2
        for i in range(m)
    )
    term2 = (1.0 - po) ** 2 * sum(
        p[i][j] * (col[i] + row[j]) ** 2
        for i in range(m)
        for j in range(m)
        if i != j
    )
    term3 = (po * pe - 2.0 * pe + po) ** 2
    var = (term1 + term2 - term3) / (n * (1.0 - pe) ** 4)
 
    se = math.sqrt(var) if var > 0 else 0.0
    result["ci95"] = (round(k - 1.96 * se, 4), round(min(1.0, k + 1.96 * se), 4))
    result["raw_agreement"] = round(po, 4)
    return result


def _rankdata(values: list[float]) -> list[float]:
    """Average-rank of values, ties share the mean rank. Local implementation to avoid a hard
    SciPy dependency for the effect-size helper."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks
