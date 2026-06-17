"""
4_analysis.py
=============

Reads the completed results CSV and produces all accuracy analysis tables
as separate CSV files in data/analysis/.

Tables produced:
  1.  summary.csv
      Overall accuracy statistics per model.

  2.  accuracy_by_language.csv
      Accuracy grouped by language.

  3.  accuracy_by_length_range.csv
      Accuracy grouped by word length range.

  4.  accuracy_by_category.csv
      Accuracy grouped by dataset category.

  5.  accuracy_by_b_count_tier.csv
      Accuracy grouped by b-count tier.

  6.  accuracy_by_distribution.csv
      Accuracy grouped by distribution pattern.

  7.  accuracy_by_language_x_length.csv
      Cross-tabulation of language × length range.

  8.  accuracy_by_language_x_category.csv
      Cross-tabulation of language × category.

  9.  accuracy_by_model_x_language_x_length.csv
      Language × length range table containing per-model statistics.

  10. consensus_errors.csv
      Cases where both models answered incorrectly.

  11. model_specific_errors.csv
      Cases where exactly one model answered incorrectly.

  12. accuracy_by_residue_class.csv
      Accuracy grouped by residue class.

  13. accuracy_by_language_x_residue_class.csv
      Cross-tabulation of language × residue class.

  14. Error-direction analysis:
        • error_direction_by_residue_class.csv
        • error_direction_by_language.csv
        • error_direction_by_language_x_residue_class.csv

      Reports false-negative (FN) and false-positive (FP) counts and rates.
      A false negative is an expected YES answered as NO.
      A false positive is an expected NO answered as YES.

  15. mod4_parity_hypothesis.csv
      Diagnostic analysis for the mod_4 language. Evaluates whether
      models appear to rely on simple parity (even/odd b-count) rather
      than true divisibility-by-4 membership by examining behavior across
      residue classes modulo 4.

All accuracy tables include:
  • total words
  • correct predictions
  • excluded answers
  • accuracy (%)

N/A and UNRESOLVABLE answers are excluded from accuracy calculations and
counted separately as excluded_count.

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
# HELPER: ERROR-DIRECTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def error_direction_stats(rows: list, group_key: callable,
                          group_label: str) -> list:
    """
    For each group, compute false-negative rate (correct=YES, model said NO)
    and false-positive rate (correct=NO, model said YES) for each model.

    A false negative here means the word belongs to the language (expected YES)
    but the model answered NO.  A false positive means the word does not belong
    (expected NO) but the model answered YES.

    Returns one row per group with counts and rates for each model.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    output = []
    for group_val in sorted(groups.keys()):
        group_rows = groups[group_val]
        entry = {group_label: group_val}

        for model in MODELS:
            valid_rows = [r for r in group_rows if is_valid(r, model)]

            yes_rows = [r for r in valid_rows if r["expected_answer"] == "YES"]
            no_rows  = [r for r in valid_rows if r["expected_answer"] == "NO"]

            # False negatives: expected YES, model said NO
            fn = sum(1 for r in yes_rows if r[f"{model}_answer"] == "NO")
            # False positives: expected NO, model said YES
            fp = sum(1 for r in no_rows  if r[f"{model}_answer"] == "YES")

            fn_rate = round(fn / len(yes_rows) * 100, 2) if yes_rows else 0.0
            fp_rate = round(fp / len(no_rows)  * 100, 2) if no_rows  else 0.0

            p = model
            entry[f"{p}_yes_total"]  = len(yes_rows)
            entry[f"{p}_fn_count"]   = fn
            entry[f"{p}_fn_rate_pct"] = fn_rate
            entry[f"{p}_no_total"]   = len(no_rows)
            entry[f"{p}_fp_count"]   = fp
            entry[f"{p}_fp_rate_pct"] = fp_rate

        output.append(entry)

    return output


def parity_hypothesis_mod4(rows: list) -> list:
    """
    Mod-4 parity hypothesis check.

    Tests whether models conflate mod-4 membership with parity (mod 2).
    Under a pure parity strategy:
      - residue 0 (b_count divisible by 4) → correctly said YES most of the time
      - residue 2 (b_count even but NOT div by 4) → incorrectly said YES
        (even count looks "correct" to a parity-only model)
      - residues 1 and 3 (odd b_count) → correctly said NO

    For each residue class within mod_4, computes:
      - accuracy (YES/NO correct)
      - false-positive rate on residue-2 words specifically
      - false-negative rate on residue-0 words specifically

    Returns one row per residue class (for mod_4 words only).
    """
    mod4_rows = [r for r in rows if r["language"] == "mod_4"]

    residue_groups = defaultdict(list)
    for row in mod4_rows:
        residue_groups[int(row["residue_class"])].append(row)

    output = []
    for residue in sorted(residue_groups.keys()):
        group_rows = residue_groups[residue]
        entry = {"language": "mod_4", "residue_class": residue}

        # Parity label: even residues (0, 2) are even b-counts
        entry["parity"] = "even" if residue % 2 == 0 else "odd"
        # True membership: only residue 0 belongs to L4
        entry["true_membership"] = "YES" if residue == 0 else "NO"

        for model in MODELS:
            valid_rows = [r for r in group_rows if is_valid(r, model)]
            if not valid_rows:
                entry[f"{model}_total"]        = 0
            else:
                correct    = sum(1 for r in valid_rows if is_correct(r, model))
                said_yes   = sum(1 for r in valid_rows if r[f"{model}_answer"] == "YES")
                said_no    = sum(1 for r in valid_rows if r[f"{model}_answer"] == "NO")
                acc        = round(correct / len(valid_rows) * 100, 2)

                entry[f"{model}_total"]        = len(valid_rows)
                entry[f"{model}_correct"]      = correct
                entry[f"{model}_accuracy_pct"] = acc
                entry[f"{model}_said_yes"]     = said_yes
                entry[f"{model}_said_no"]      = said_no
                entry[f"{model}_said_yes_pct"] = round(said_yes / len(valid_rows) * 100, 2)

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

def error_direction_fields() -> list:
    """Return column names for error-direction tables."""
    fields = []
    for model in MODELS:
        fields += [
            f"{model}_yes_total", f"{model}_fn_count", f"{model}_fn_rate_pct",
            f"{model}_no_total",  f"{model}_fp_count", f"{model}_fp_rate_pct",
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
    ERROR_FIELDS = [
        "word", "language", "length", "length_range", "b_count", "b_count_tier",
        "distribution_pattern", "residue_class", "category", "expected_answer",
        "chatgpt_answer", "chatgpt_correct", "chatgpt_runs",
        "gemini_answer", "gemini_correct", "gemini_runs",
        "batch_id",
    ]
    write_table(
        os.path.join(ANALYSIS_DIR, "consensus_errors.csv"),
        ERROR_FIELDS,
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
        ERROR_FIELDS + ["wrong_model"],
        model_specific,
    )
    print(f"    Model-specific errors: {len(model_specific)}")

    # ── 12. Accuracy by residue class (all languages combined) ────────────────
    print("Computing accuracy by residue class...")
    table = grouped_accuracy(
        rows, lambda r: int(r["residue_class"]), "residue_class"
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_residue_class.csv"),
        ["residue_class"] + stat_fields, table,
    )

    # ── 13. Cross: language × residue class ───────────────────────────────────
    print("Computing language × residue class cross tabulation...")
    table = cross_accuracy(
        rows,
        lambda r: r["language"],
        lambda r: int(r["residue_class"]),
        "language", "residue_class",
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "accuracy_by_language_x_residue_class.csv"),
        ["language", "residue_class"] + stat_fields, table,
    )

    # ── 14. Error direction: false-negative and false-positive rates ──────────
    #        Grouped by expected answer × residue class, then by language
    print("Computing error-direction analysis (FN/FP rates)...")
    ed_fields = error_direction_fields()

    # 14a. Overall FN/FP by residue class
    table = error_direction_stats(
        rows, lambda r: int(r["residue_class"]), "residue_class"
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "error_direction_by_residue_class.csv"),
        ["residue_class"] + ed_fields, table,
    )

    # 14b. FN/FP by language (are certain languages driving false-negative bias?)
    table = error_direction_stats(
        rows, lambda r: r["language"], "language"
    )
    write_table(
        os.path.join(ANALYSIS_DIR, "error_direction_by_language.csv"),
        ["language"] + ed_fields, table,
    )

    # 14c. FN/FP by language × residue class (the most diagnostic table)
    print("Computing language × residue class error-direction cross tabulation...")

    lang_res_groups = defaultdict(list)
    for row in rows:
        lang_res_groups[(row["language"], int(row["residue_class"]))].append(row)

    ed_cross_rows = []
    for (lang, res) in sorted(lang_res_groups.keys()):
        group_rows = lang_res_groups[(lang, res)]
        sub = error_direction_stats(
            group_rows, lambda r, _lang=lang, _res=res: f"{_lang}_r{_res}", "key"
        )
        if sub:
            entry = {"language": lang, "residue_class": res}
            entry.update({k: v for k, v in sub[0].items() if k != "key"})
            ed_cross_rows.append(entry)

    write_table(
        os.path.join(ANALYSIS_DIR, "error_direction_by_language_x_residue_class.csv"),
        ["language", "residue_class"] + ed_fields, ed_cross_rows,
    )

    # ── 15. Mod-4 parity hypothesis check ────────────────────────────────────
    print("Computing mod-4 parity hypothesis check...")
    parity_fields = ["language", "residue_class", "parity", "true_membership"]
    for model in MODELS:
        parity_fields += [
            f"{model}_total", f"{model}_correct", f"{model}_accuracy_pct",
            f"{model}_said_yes", f"{model}_said_no", f"{model}_said_yes_pct",
        ]
    table = parity_hypothesis_mod4(rows)
    write_table(
        os.path.join(ANALYSIS_DIR, "mod4_parity_hypothesis.csv"),
        parity_fields, table,
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\nAll analysis tables written to {ANALYSIS_DIR}/")
    print("You can now import these CSVs to build charts manually.")


if __name__ == "__main__":
    main()
