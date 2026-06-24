"""
4_analysis.py  (v2 — adds statistical tests + fixes three bugs in run_all_tests)
=================================================================================
Reads the completed results CSV and produces all accuracy analysis tables
as separate CSV files in data/analysis/.

NEW in this version:
  * chi_square_tests.csv  — χ² test + p-value for every contingency table
  * linear_trend_tests.csv — linear-by-linear (Cochran–Armitage) trend test
    for ordered groupings: length_range and b_count_tier
  These results are also printed to stdout for easy copy-paste into LaTeX.

Tables produced:
  1.  accuracy_by_language.csv
  2.  accuracy_by_length_range.csv          + trend test
  3.  accuracy_by_category.csv
  4.  accuracy_by_b_count_tier.csv          + trend test
  5.  accuracy_by_distribution.csv          + χ² test
  6.  accuracy_by_language_x_length.csv     + χ² test (cross tabulation)
  7.  accuracy_by_language_x_category.csv   + χ² test (cross tabulation)
  8.  accuracy_by_model_x_language_x_length.csv
  9.  consensus_errors.csv
  10. model_specific_errors.csv
  11. summary.csv
  12. chi_square_tests.csv                  (NEW)
  13. linear_trend_tests.csv                (NEW)

Run: pip install scipy --break-system-packages
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE = "data/results.csv"
ANALYSIS_DIR = "data/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Ordered groupings (for trend tests) ──────────────────────────────────────
LENGTH_RANGES = [
    (1,  15,  "1-15"),
    (16, 30,  "16-30"),
    (31, 60,  "31-60"),
    (61, 100, "61-100"),
]
LENGTH_RANGE_ORDER = ["1-15", "16-30", "31-60", "61-100"]
B_COUNT_TIER_ORDER = ["very_low", "low", "medium", "high", "very_high"]

MODELS = ["chatgpt", "gemini"]

# Significance threshold
ALPHA = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — loading & basic stats
# ══════════════════════════════════════════════════════════════════════════════

def length_range_label(length: int) -> str:
    for lo, hi, label in LENGTH_RANGES:
        if lo <= length <= hi:
            return label
    return "unknown"


def load_results(filepath: str) -> list:
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["length"] = int(row["length"])
            row["b_count"] = int(row["b_count"])
            row["length_range"] = length_range_label(row["length"])
            rows.append(row)
    return rows


def is_valid(row: dict, model: str) -> bool:
    return row[f"{model}_answer"] in ("YES", "NO")


def is_correct(row: dict, model: str) -> bool:
    return row[f"{model}_correct"] == "TRUE"


def accuracy_stats(rows: list, model: str) -> dict:
    total   = len(rows)
    valid   = [r for r in rows if is_valid(r, model)]
    correct = [r for r in valid if is_correct(r, model)]
    excluded = total - len(valid)
    acc = (len(correct) / len(valid) * 100) if valid else 0.0
    return {
        "total": total, "valid": len(valid), "correct": len(correct),
        "excluded": excluded, "accuracy_pct": round(acc, 2),
    }


def write_table(filepath: str, fieldnames: list, rows: list):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"   # <-- add this
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {filepath}")


def model_stat_fields() -> list:
    fields = []
    for model in MODELS:
        fields += [f"{model}_total", f"{model}_correct",
                   f"{model}_excluded", f"{model}_accuracy_pct"]
    return fields


def grouped_accuracy(rows, group_key, group_label):
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    output = []
    for gval in sorted(groups.keys()):
        grow = groups[gval]
        entry = {group_label: gval}
        for model in MODELS:
            s = accuracy_stats(grow, model)
            entry[f"{model}_total"]        = s["total"]
            entry[f"{model}_correct"]      = s["correct"]
            entry[f"{model}_excluded"]     = s["excluded"]
            entry[f"{model}_accuracy_pct"] = s["accuracy_pct"]
        output.append(entry)
    return output


def cross_accuracy(rows, key1, key2, label1, label2):
    groups = defaultdict(list)
    for row in rows:
        groups[(key1(row), key2(row))].append(row)
    output = []
    for (v1, v2) in sorted(groups.keys()):
        grow = groups[(v1, v2)]
        entry = {label1: v1, label2: v2}
        for model in MODELS:
            s = accuracy_stats(grow, model)
            entry[f"{model}_total"]        = s["total"]
            entry[f"{model}_correct"]      = s["correct"]
            entry[f"{model}_excluded"]     = s["excluded"]
            entry[f"{model}_accuracy_pct"] = s["accuracy_pct"]
        output.append(entry)
    return output


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def pvalue_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def chi2_test_for_groups(rows: list, group_key, model: str,
                          group_order: list = None) -> dict:
    """
    Build a 2×N contingency table (correct vs. incorrect) over groups,
    run a Pearson χ² test, and return the result dict.

    group_order: if provided, fixes the column order (for trend tests too).
    """
    groups = defaultdict(list)
    for row in rows:
        if is_valid(row, model):
            groups[group_key(row)].append(row)

    keys = group_order if group_order else sorted(groups.keys())
    # Build 2×N matrix: row0 = correct, row1 = incorrect
    matrix = []
    for k in keys:
        if k not in groups:
            continue
        grow = groups[k]
        correct   = sum(1 for r in grow if is_correct(r, model))
        incorrect = len(grow) - correct
        matrix.append([correct, incorrect])

    if len(matrix) < 2:
        return {"chi2": None, "p": None, "dof": None, "note": "insufficient groups"}

    table = np.array(matrix).T  # shape: 2 × N_groups
    chi2, p, dof, expected = chi2_contingency(table)
    return {
        "chi2": round(float(chi2), 4),
        "p":    round(float(p), 6),
        "dof":  int(dof),
        "sig":  pvalue_stars(p),
        "note": "",
    }


def linear_by_linear_test(rows: list, group_key, model: str,
                           ordered_levels: list) -> dict:
    """
    Linear-by-linear association (Cochran–Armitage trend test).

    Assigns integer scores 1,2,3,… to ordered_levels and computes:
        M² = (n-1) · r²   where r is the Pearson correlation between
        the score and the binary correct/incorrect outcome.
    Distributed as χ²(1) under H₀.

    This tests whether accuracy monotonically increases or decreases
    across the ordered grouping.
    """
    scores_list   = []
    outcomes_list = []
    for row in rows:
        if not is_valid(row, model):
            continue
        key = group_key(row)
        if key not in ordered_levels:
            continue
        score = ordered_levels.index(key) + 1   # 1-based integer score
        outcome = 1 if is_correct(row, model) else 0
        scores_list.append(score)
        outcomes_list.append(outcome)

    if len(scores_list) < 10:
        return {"M2": None, "p": None, "note": "insufficient data"}

    x = np.array(scores_list,   dtype=float)
    y = np.array(outcomes_list, dtype=float)
    n = len(x)

    r, p_pearson = stats.pearsonr(x, y)
    # M² statistic = (n-1)*r²,  df=1
    M2 = (n - 1) * r ** 2
    p_val = stats.chi2.sf(M2, df=1)

    direction = "decreasing" if r < 0 else "increasing"
    return {
        "r":         round(float(r), 4),
        "M2":        round(float(M2), 4),
        "p":         round(float(p_val), 6),
        "sig":       pvalue_stars(p_val),
        "direction": direction,
        "note":      "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# COLLECT AND PRINT ALL TEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests(rows: list) -> tuple:
    """
    Run all χ² and linear-by-linear tests.
    Returns (chi2_results, trend_results) as lists of dicts.

    Bug fixes applied:
      Bug 1 — per-language loops now receive an explicit `subset` argument so
               the closure does not silently fall back to all 3,000 rows.
      Bug 2 — L4 parity test explicitly passes l4_rows instead of all rows.
      Bug 3 — duplicate distribution_pattern_overall test removed.
    """
    chi2_results  = []
    trend_results = []

    # ── Helper to register a chi2 test ────────────────────────────────────────
    # `subset` is the row-list to use; defaults to the full `rows` when omitted.
    def add_chi2(label, group_key, group_order=None, subset=None):
        data = subset if subset is not None else rows
        for model in MODELS:
            res = chi2_test_for_groups(data, group_key, model, group_order)
            chi2_results.append({
                "test":  label,
                "model": model,
                "chi2":  res.get("chi2"),
                "dof":   res.get("dof"),
                "p":     res.get("p"),
                "sig":   res.get("sig", ""),
                "note":  res.get("note", ""),
            })

    # ── Helper to register a trend test ───────────────────────────────────────
    def add_trend(label, group_key, ordered_levels, subset=None):
        data = subset if subset is not None else rows
        for model in MODELS:
            res = linear_by_linear_test(data, group_key, model, ordered_levels)
            trend_results.append({
                "test":      label,
                "model":     model,
                "r":         res.get("r"),
                "M2":        res.get("M2"),
                "p":         res.get("p"),
                "sig":       res.get("sig", ""),
                "direction": res.get("direction", ""),
                "note":      res.get("note", ""),
            })

    # 1. Accuracy by language (χ²)  — all rows
    add_chi2("accuracy_by_language",
             lambda r: r["language"])

    # 2. Accuracy by length range (χ² + trend)  — all rows
    add_chi2("accuracy_by_length_range",
             lambda r: r["length_range"],
             group_order=LENGTH_RANGE_ORDER)
    add_trend("trend_accuracy_by_length_range",
              lambda r: r["length_range"],
              LENGTH_RANGE_ORDER)

    # 3. Accuracy by category (χ²)  — all rows
    add_chi2("accuracy_by_category",
             lambda r: r["category"])

    # 4. Accuracy by b_count_tier (χ² + trend)  — all rows
    add_chi2("accuracy_by_b_count_tier",
             lambda r: r["b_count_tier"],
             group_order=B_COUNT_TIER_ORDER)
    add_trend("trend_accuracy_by_b_count_tier",
              lambda r: r["b_count_tier"],
              B_COUNT_TIER_ORDER)

    # 5. Accuracy by distribution pattern (χ²)  — all rows
    # Bug 3 fixed: only one distribution-pattern test (duplicate removed)
    add_chi2("accuracy_by_distribution_pattern",
             lambda r: r["distribution_pattern"])

    # 6. Cross: language × length range — per-language subsets
    # Bug 1 fixed: lang_rows is passed explicitly via `subset=`
    # NOTE: language column uses "mod_3"/"mod_4"/"mod_5" in the CSV, not "L3"/"L4"/"L5"
    for lang, label in [("mod_3", "L3"), ("mod_4", "L4"), ("mod_5", "L5")]:
        lang_rows = [r for r in rows if r["language"] == lang]
        add_chi2(f"language_x_length__{label}",
                 lambda r: r["length_range"],
                 group_order=LENGTH_RANGE_ORDER,
                 subset=lang_rows)
        add_trend(f"trend_language_x_length__{label}",
                  lambda r: r["length_range"],
                  LENGTH_RANGE_ORDER,
                  subset=lang_rows)

    # 7. Cross: language × category — per-language subsets
    # Bug 1 fixed: lang_rows is passed explicitly via `subset=`
    for lang, label in [("mod_3", "L3"), ("mod_4", "L4"), ("mod_5", "L5")]:
        lang_rows = [r for r in rows if r["language"] == lang]
        add_chi2(f"language_x_category__{label}",
                 lambda r: r["category"],
                 subset=lang_rows)

    # 8. Parity hypothesis for L4 — L4 rows only
    # Bug 2 fixed: l4_rows passed explicitly via `subset=` so L3/L5 residues
    # (which overlap with L4's r∈{0,1,2,3}) are no longer included.
    # residue_class is stored as a string digit in the CSV ("0","1","2","3").
    l4_rows = [r for r in rows if r["language"] == "mod_4"]
    add_chi2("L4_parity_hypothesis",
             lambda r: r["residue_class"],
             group_order=["0", "1", "2", "3"],
             subset=l4_rows)

    return chi2_results, trend_results


def print_test_summary(chi2_results, trend_results):
    """Print a human-readable summary to stdout for easy LaTeX copy-paste."""
    sep = "─" * 70
    print(f"\n{'═'*70}")
    print("  STATISTICAL TEST RESULTS")
    print(f"{'═'*70}\n")

    print("CHI-SQUARE TESTS  (H₀: accuracy is independent of grouping)")
    print(sep)
    print(f"{'Test':<42} {'Model':<10} {'χ²':>8} {'df':>4} {'p':>9} {'Sig':>4}")
    print(sep)
    for r in chi2_results:
        chi2_str = f"{r['chi2']:.4f}" if r['chi2'] is not None else "N/A"
        p_str    = f"{r['p']:.6f}"    if r['p']    is not None else "N/A"
        dof_str  = str(r['dof'])      if r['dof']  is not None else "N/A"
        print(f"{r['test']:<42} {r['model']:<10} {chi2_str:>8} {dof_str:>4} "
              f"{p_str:>9} {r.get('sig',''):>4}")
    print()

    print("LINEAR-BY-LINEAR TREND TESTS  (H₀: no monotone trend)")
    print(sep)
    print(f"{'Test':<42} {'Model':<10} {'r':>7} {'M²':>8} {'p':>9} "
          f"{'Sig':>4} {'Direction'}")
    print(sep)
    for r in trend_results:
        r_str  = f"{r['r']:.4f}"  if r['r']  is not None else "N/A"
        m2_str = f"{r['M2']:.4f}" if r['M2'] is not None else "N/A"
        p_str  = f"{r['p']:.6f}"  if r['p']  is not None else "N/A"
        print(f"{r['test']:<42} {r['model']:<10} {r_str:>7} {m2_str:>8} "
              f"{p_str:>9} {r.get('sig',''):>4}  {r.get('direction','')}")
    print()
    print("Significance codes:  *** p<0.001   ** p<0.01   * p<0.05   ns p≥0.05")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading results from {RESULTS_FILE}...")
    rows = load_results(RESULTS_FILE)
    print(f"Loaded {len(rows)} rows.\n")

    stat_fields = model_stat_fields()

    # ── 1. Overall summary ────────────────────────────────────────────────────
    print("Computing summary...")
    summary_rows = []
    for model in MODELS:
        s = accuracy_stats(rows, model)
        summary_rows.append({"model": model, **s})
    write_table(
        os.path.join(ANALYSIS_DIR, "summary.csv"),
        ["model", "total", "valid", "correct", "excluded", "accuracy_pct"],
        summary_rows,
    )

    # ── 2. Accuracy by language ───────────────────────────────────────────────
    print("Computing accuracy by language...")
    table = grouped_accuracy(rows, lambda r: r["language"], "language")
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_language.csv"),
        ["language"] + stat_fields, table,
    )

    # ── 3. Accuracy by length range ───────────────────────────────────────────
    print("Computing accuracy by length range...")
    table = grouped_accuracy(rows, lambda r: r["length_range"], "length_range")
    table.sort(key=lambda x: LENGTH_RANGE_ORDER.index(x["length_range"])
               if x["length_range"] in LENGTH_RANGE_ORDER else 99)
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_length_range.csv"),
        ["length_range"] + stat_fields, table,
    )

    # ── 4. Accuracy by category ───────────────────────────────────────────────
    print("Computing accuracy by category...")
    table = grouped_accuracy(rows, lambda r: r["category"], "category")
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_category.csv"),
        ["category"] + stat_fields, table,
    )

    # ── 5. Accuracy by b-count tier ───────────────────────────────────────────
    print("Computing accuracy by b-count tier...")
    table = grouped_accuracy(rows, lambda r: r["b_count_tier"], "b_count_tier")
    table.sort(key=lambda x: B_COUNT_TIER_ORDER.index(x["b_count_tier"])
               if x["b_count_tier"] in B_COUNT_TIER_ORDER else 99)
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_b_count_tier.csv"),
        ["b_count_tier"] + stat_fields, table,
    )

    # ── 6. Accuracy by distribution pattern ───────────────────────────────────
    print("Computing accuracy by distribution pattern...")
    table = grouped_accuracy(
        rows, lambda r: r["distribution_pattern"], "distribution_pattern"
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_distribution.csv"),
        ["distribution_pattern"] + stat_fields, table,
    )

    # ── 7. Cross: language × length range ─────────────────────────────────────
    print("Computing language × length range cross tabulation...")
    table = cross_accuracy(
        rows, lambda r: r["language"], lambda r: r["length_range"],
        "language", "length_range",
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_language_x_length.csv"),
        ["language", "length_range"] + stat_fields, table,
    )

    # ── 8. Cross: language × category ─────────────────────────────────────────
    print("Computing language × category cross tabulation...")
    table = cross_accuracy(
        rows, lambda r: r["language"], lambda r: r["category"],
        "language", "category",
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_language_x_category.csv"),
        ["language", "category"] + stat_fields, table,
    )

    # ── 9. Cross: model × language × length ───────────────────────────────────
    print("Computing model × language × length cross tabulation...")
    table = cross_accuracy(
        rows, lambda r: r["language"], lambda r: r["length_range"],
        "language", "length_range",
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_model_x_language_x_length.csv"),
        ["language", "length_range"] + stat_fields, table,
    )

    # ── 10. Consensus errors ──────────────────────────────────────────────────
    print("Computing consensus errors (both models wrong)...")
    consensus_errors = [
        r for r in rows
        if is_valid(r, "chatgpt") and is_valid(r, "gemini")
        and not is_correct(r, "chatgpt") and not is_correct(r, "gemini")
    ]
    # Use all columns present in the enriched row dicts (includes length_range added at load)
    FULL_FIELDS = [
        "word", "language", "length", "b_count", "b_count_tier",
        "distribution_pattern", "residue_class", "category",
        "expected_answer", "batch_id", "chatgpt_answer", "chatgpt_correct",
        "gemini_answer", "gemini_correct", "chatgpt_runs", "gemini_runs",
        "length_range",
    ]
    write_table(
        os.path.join(ANALYSIS_DIR, "consensus_errors.csv"),
        FULL_FIELDS,
        consensus_errors,
    )
    print(f"    Consensus errors: {len(consensus_errors)}")

    # ── 11. Model-specific errors ─────────────────────────────────────────────
    print("Computing model-specific errors (only one model wrong)...")
    model_specific = []
    for r in rows:
        if not (is_valid(r, "chatgpt") and is_valid(r, "gemini")):
            continue
        chatgpt_ok = is_correct(r, "chatgpt")
        gemini_ok  = is_correct(r, "gemini")
        if chatgpt_ok != gemini_ok:
            entry = dict(r)
            entry["wrong_model"] = "chatgpt" if not chatgpt_ok else "gemini"
            model_specific.append(entry)
    write_table(
        os.path.join(ANALYSIS_DIR, "model_specific_errors.csv"),
        FULL_FIELDS + ["wrong_model"],
        model_specific,
    )
    print(f"    Model-specific errors: {len(model_specific)}")

    # ── 12 & 13. Statistical tests ────────────────────────────────────────────
    print("\nRunning statistical tests...")
    chi2_results, trend_results = run_all_tests(rows)

    write_table(
        os.path.join(ANALYSIS_DIR, "chi_square_tests.csv"),
        ["test", "model", "chi2", "dof", "p", "sig", "note"],
        chi2_results,
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "linear_trend_tests.csv"),
        ["test", "model", "r", "M2", "p", "sig", "direction", "note"],
        trend_results,
    )

    # Print human-readable summary for LaTeX copy-paste
    print_test_summary(chi2_results, trend_results)

    print(f"\nAll analysis tables written to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
