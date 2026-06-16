"""
4_analysis.py
=============
Reads the completed results CSV and produces all accuracy analysis tables
as separate CSV files in data/analysis/.

Tables produced:
  1.  accuracy_by_language.csv
  2.  accuracy_by_length_range.csv
  3.  accuracy_by_category.csv
  4.  accuracy_by_b_count_tier.csv
  5.  accuracy_by_distribution.csv
  6.  accuracy_by_language_x_length.csv       (cross tabulation)
  7.  accuracy_by_language_x_category.csv     (cross tabulation)
  8.  accuracy_by_model_x_language_x_length.csv (cross tabulation)
  9.  consensus_errors.csv                    (both models wrong)
  10. model_specific_errors.csv               (only one model wrong)
  11. summary.csv                             (overall accuracy per model)

All tables include: total words, correct predictions, accuracy (%).
N/A and UNRESOLVABLE answers are excluded from calculations and counted
separately as excluded_count.

Run this script after all model responses have been recorded.
"""

import csv
import os
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE  = "data/results.csv"
ANALYSIS_DIR  = "data/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Length range labels (must match generator) ────────────────────────────────
LENGTH_RANGES = [
    (1,  15,  "1-15"),
    (16, 30,  "16-30"),
    (31, 60,  "31-60"),
    (61, 100, "61-100"),
]

MODELS = ["chatgpt", "gemini"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def length_range_label(length: int) -> str:
    """Return the length range label for a given word length."""
    for lo, hi, label in LENGTH_RANGES:
        if lo <= length <= hi:
            return label
    return "unknown"


def load_results(filepath: str) -> list:
    """Load results CSV into a list of dicts."""
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
    """Return True if this row has a usable (non-excluded) answer."""
    ans = row[f"{model}_answer"]
    return ans in ("YES", "NO")


def is_correct(row: dict, model: str) -> bool:
    """Return True if the model answered correctly."""
    return row[f"{model}_correct"] == "TRUE"


def accuracy_stats(rows: list, model: str) -> dict:
    """
    Compute accuracy statistics for a list of rows and one model.

    Returns dict with: total, valid, correct, excluded, accuracy_pct
    """
    total    = len(rows)
    valid    = [r for r in rows if is_valid(r, model)]
    correct  = [r for r in valid if is_correct(r, model)]
    excluded = total - len(valid)
    acc      = (len(correct) / len(valid) * 100) if valid else 0.0

    return {
        "total":        total,
        "valid":        len(valid),
        "correct":      len(correct),
        "excluded":     excluded,
        "accuracy_pct": round(acc, 2),
    }


def write_table(filepath: str, fieldnames: list, rows: list):
    """Write a list of dicts to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {filepath}")


def grouped_accuracy(rows: list, group_key: callable,
                     group_label: str) -> list:
    """
    Compute per-model accuracy for groups defined by group_key(row).

    Args:
        rows        : all result rows
        group_key   : function that takes a row and returns a group label
        group_label : column name for the group in the output table

    Returns:
        List of dicts suitable for write_table.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    output = []
    for group_val in sorted(groups.keys()):
        group_rows = groups[group_val]
        entry = {group_label: group_val}
        for model in MODELS:
            stats = accuracy_stats(group_rows, model)
            prefix = model
            entry[f"{prefix}_total"]        = stats["total"]
            entry[f"{prefix}_correct"]      = stats["correct"]
            entry[f"{prefix}_excluded"]     = stats["excluded"]
            entry[f"{prefix}_accuracy_pct"] = stats["accuracy_pct"]
        output.append(entry)

    return output


def cross_accuracy(rows: list, key1: callable, key2: callable,
                   label1: str, label2: str) -> list:
    """
    Compute per-model accuracy for cross-tabulation of two grouping keys.

    Returns:
        List of dicts with label1, label2, and per-model stats.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(key1(row), key2(row))].append(row)

    output = []
    for (v1, v2) in sorted(groups.keys()):
        group_rows = groups[(v1, v2)]
        entry = {label1: v1, label2: v2}
        for model in MODELS:
            stats = accuracy_stats(group_rows, model)
            prefix = model
            entry[f"{prefix}_total"]        = stats["total"]
            entry[f"{prefix}_correct"]      = stats["correct"]
            entry[f"{prefix}_excluded"]     = stats["excluded"]
            entry[f"{prefix}_accuracy_pct"] = stats["accuracy_pct"]
        output.append(entry)

    return output


# ══════════════════════════════════════════════════════════════════════════════
# TABLE FIELD NAME BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def model_stat_fields() -> list:
    """Return the per-model stat column names."""
    fields = []
    for model in MODELS:
        fields += [
            f"{model}_total", f"{model}_correct",
            f"{model}_excluded", f"{model}_accuracy_pct",
        ]
    return fields


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
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
        stats = accuracy_stats(rows, model)
        summary_rows.append({"model": model, **stats})
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
    tier_order = ["very_low", "low", "medium", "high", "very_high"]
    table = grouped_accuracy(rows, lambda r: r["b_count_tier"], "b_count_tier")
    # Sort by defined tier order
    table.sort(key=lambda x: tier_order.index(x["b_count_tier"])
               if x["b_count_tier"] in tier_order else 99)
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
        rows,
        lambda r: r["language"],
        lambda r: r["length_range"],
        "language", "length_range",
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_language_x_length.csv"),
        ["language", "length_range"] + stat_fields, table,
    )

    # ── 8. Cross: language × category ─────────────────────────────────────────
    print("Computing language × category cross tabulation...")
    table = cross_accuracy(
        rows,
        lambda r: r["language"],
        lambda r: r["category"],
        "language", "category",
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_language_x_category.csv"),
        ["language", "category"] + stat_fields, table,
    )

    # ── 9. Cross: model × language × length (three-way) ──────────────────────
    print("Computing model × language × length cross tabulation...")
    # For this one we pivot differently — one row per (language, length_range),
    # with columns for each model's accuracy
    table = cross_accuracy(
        rows,
        lambda r: r["language"],
        lambda r: r["length_range"],
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
    write_table(
        os.path.join(ANALYSIS_DIR, "consensus_errors.csv"),
        [
            "word", "language", "length", "b_count", "b_count_tier",
            "distribution_pattern", "residue_class", "category",
            "expected_answer", "chatgpt_answer", "gemini_answer", "batch_id",
        ],
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
        [
            "word", "language", "length", "b_count", "b_count_tier",
            "distribution_pattern", "residue_class", "category",
            "expected_answer", "chatgpt_answer", "gemini_answer",
            "wrong_model", "batch_id",
        ],
        model_specific,
    )
    print(f"    Model-specific errors: {len(model_specific)}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\nAll analysis tables written to {ANALYSIS_DIR}/")
    print("You can now import these CSVs to build charts manually.")


if __name__ == "__main__":
    main()
