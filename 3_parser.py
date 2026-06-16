"""
3_parser.py
===========
Parses model responses (ChatGPT and Gemini) and records results into the
master results CSV.

WORKFLOW
--------
1. Run this script once per batch per model.
2. When prompted, paste the model's raw response and press Enter twice.
3. The script parses each line, matches words, and records answers.
4. Flagged lines (MISMATCH / MALFORMED / INVALID / MISSING) are reported
   so you know which batches need re-running.
5. Results are saved incrementally — you can stop and resume at any time.

Re-run policy:
  - If a batch has ANY flagged responses, re-run the ENTIRE batch in a
    fresh session (do not continue in the same conversation).
  - Maximum 3 re-run attempts per batch per model.
  - If still unresolvable after 3 attempts, the words are marked UNRESOLVABLE
    and excluded from accuracy calculations.

Output:
  data/results.csv   — master results file (created on first run)
"""

import csv
import os
import re
import sys
from datetime import date

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_SET_FILE = "data/test_set.csv"
RESULTS_FILE  = "data/results.csv"

# ── Valid models ──────────────────────────────────────────────────────────────
VALID_MODELS = ["chatgpt", "gemini"]

# ── Result field names ────────────────────────────────────────────────────────
# Original test set fields + one answer column per model
BASE_FIELDS = [
    "word", "language", "length", "b_count", "b_count_tier",
    "distribution_pattern", "residue_class", "category",
    "expected_answer", "batch_id",
]
RESULT_FIELDS = BASE_FIELDS + [
    "chatgpt_answer", "chatgpt_correct",
    "gemini_answer",  "gemini_correct",
    "chatgpt_runs",   "gemini_runs",   # how many attempts were needed
]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD / INITIALISE RESULTS FILE
# ══════════════════════════════════════════════════════════════════════════════

def load_results() -> dict:
    """
    Load existing results file into memory as a dict keyed by word+language.
    If the file doesn't exist, initialise it from the test set.

    Returns:
        dict mapping (word, language) -> result record dict
    """
    if os.path.exists(RESULTS_FILE):
        records = {}
        with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["word"], row["language"])
                records[key] = row
        print(f"Loaded {len(records)} existing results from {RESULTS_FILE}")
        return records

    # Initialise from test set
    print(f"No results file found. Initialising from {TEST_SET_FILE}...")
    records = {}
    with open(TEST_SET_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["word"], row["language"])
            record = {field: row.get(field, "") for field in BASE_FIELDS}
            record["chatgpt_answer"]  = ""
            record["chatgpt_correct"] = ""
            record["gemini_answer"]   = ""
            record["gemini_correct"]  = ""
            record["chatgpt_runs"]    = "0"
            record["gemini_runs"]     = "0"
            records[key] = record
    save_results(records)
    print(f"Initialised results file at {RESULTS_FILE}")
    return records


def save_results(records: dict):
    """Write all results back to the CSV file."""
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(records.values())


# ══════════════════════════════════════════════════════════════════════════════
# LOAD BATCH WORD ORDER
# ══════════════════════════════════════════════════════════════════════════════

def load_batch(batch_id: str) -> list:
    """
    Return the ordered list of words for a given batch_id from the test set.

    Args:
        batch_id : e.g. 'mod3_batch001'

    Returns:
        List of (word, language) tuples in batch order.
    """
    words = []
    with open(TEST_SET_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["batch_id"] == batch_id:
                words.append((row["word"], row["language"]))
    return words


# ══════════════════════════════════════════════════════════════════════════════
# PARSE MODEL RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

def parse_response(raw: str, expected_words: list) -> list:
    """
    Parse a raw model response string into a list of result dicts.

    Expected format per line:
        N. [word] : YES
        N. [word] : NO

    Each result dict contains:
        word        : word from the response line
        answer      : 'YES', 'NO', or a flag
        flag        : None if clean, else 'MISMATCH'/'MALFORMED'/'INVALID'/'MISSING'
        line        : original response line for debugging

    Args:
        raw            : full raw response string from the model
        expected_words : ordered list of (word, language) tuples for this batch

    Returns:
        List of result dicts, one per expected word.
    """
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    results = []

    for i, (exp_word, exp_lang) in enumerate(expected_words):
        if i >= len(lines):
            results.append({
                "word": exp_word, "language": exp_lang,
                "answer": "MISSING", "flag": "MISSING", "line": ""
            })
            continue

        line = lines[i]

        # Try to parse "N. word : YES/NO"
        match = re.match(
            r"^\d+\.\s*(.+?)\s*:\s*(yes|no)\s*$",
            line,
            re.IGNORECASE
        )

        if not match:
            results.append({
                "word": exp_word, "language": exp_lang,
                "answer": "MALFORMED", "flag": "MALFORMED", "line": line
            })
            continue

        found_word = match.group(1).strip()
        found_ans  = match.group(2).strip().upper()

        if found_word != exp_word:
            results.append({
                "word": exp_word, "language": exp_lang,
                "answer": "MISMATCH", "flag": "MISMATCH", "line": line
            })
            continue

        if found_ans not in ("YES", "NO"):
            results.append({
                "word": exp_word, "language": exp_lang,
                "answer": "INVALID", "flag": "INVALID", "line": line
            })
            continue

        results.append({
            "word": exp_word, "language": exp_lang,
            "answer": found_ans, "flag": None, "line": line
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# RECORD RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def record_results(parsed: list, model: str, records: dict,
                   run_number: int) -> bool:
    """
    Write parsed results into the master records dict.

    Args:
        parsed     : list of result dicts from parse_response()
        model      : 'chatgpt' or 'gemini'
        records    : master results dict (mutated in place)
        run_number : which attempt this is (1, 2, or 3)

    Returns:
        True if all results are clean (no flags), False otherwise.
    """
    all_clean = True

    for res in parsed:
        key = (res["word"], res["language"])
        if key not in records:
            print(f"  WARNING: word not found in test set: {res['word']}")
            continue

        record = records[key]
        answer = res["answer"]
        flag   = res["flag"]

        if flag:
            all_clean = False
            print(f"  FLAG [{flag}] line: '{res['line']}'  expected: '{res['word']}'")

        # Only write if this is a clean answer OR it's the final run
        record[f"{model}_answer"]  = answer
        record[f"{model}_correct"] = (
            "TRUE"  if answer == record["expected_answer"] else
            "FALSE" if answer in ("YES", "NO") else
            "N/A"
        )
        record[f"{model}_runs"] = str(run_number)

    return all_clean


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE SESSION
# ══════════════════════════════════════════════════════════════════════════════

def get_multiline_input(prompt: str) -> str:
    """Read multi-line input from the user until two consecutive blank lines."""
    print(prompt)
    print("(Paste the response, then press Enter twice to finish)\n")
    lines = []
    blank_count = 0
    while True:
        line = input()
        if line == "":
            blank_count += 1
            if blank_count >= 2:
                break
        else:
            blank_count = 0
        lines.append(line)
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  RESPONSE PARSER — Formal Language LLM Experiment")
    print("=" * 60)

    # Load or initialise results
    records = load_results()

    # Select model
    print("\nWhich model are you recording responses for?")
    print("  1. chatgpt")
    print("  2. gemini")
    model_choice = input("Enter 1 or 2: ").strip()
    model = "chatgpt" if model_choice == "1" else "gemini"
    print(f"Selected model: {model}")

    # Select batch
    batch_id = input("\nEnter batch ID (e.g. mod3_batch001): ").strip()

    # Load expected words for this batch
    expected_words = load_batch(batch_id)
    if not expected_words:
        print(f"ERROR: batch '{batch_id}' not found in test set.")
        sys.exit(1)
    print(f"Loaded {len(expected_words)} words for batch {batch_id}.")

    # Check if already recorded
    sample_key  = expected_words[0]
    sample_rec  = records.get(sample_key, {})
    prev_runs   = int(sample_rec.get(f"{model}_runs", "0"))
    prev_answer = sample_rec.get(f"{model}_answer", "")

    if prev_answer and prev_answer not in ("MISSING", "MALFORMED",
                                            "MISMATCH", "INVALID",
                                            "UNRESOLVABLE", ""):
        print(f"\nThis batch already has clean results for {model}.")
        redo = input("Re-record anyway? (y/n): ").strip().lower()
        if redo != "y":
            print("Skipping.")
            return

    run_number = prev_runs + 1
    print(f"\nThis is run attempt #{run_number} for {model} / {batch_id}.")

    if run_number > 3:
        print("\nMaximum re-run attempts (3) reached for this batch.")
        print("Marking remaining flagged words as UNRESOLVABLE.")
        for word, lang in expected_words:
            key = (word, lang)
            rec = records[key]
            if rec[f"{model}_answer"] in ("MISSING", "MALFORMED",
                                           "MISMATCH", "INVALID", ""):
                rec[f"{model}_answer"]  = "UNRESOLVABLE"
                rec[f"{model}_correct"] = "N/A"
                rec[f"{model}_runs"]    = "3"
        save_results(records)
        print("Saved. These words will be excluded from accuracy calculations.")
        return

    # Get response
    raw_response = get_multiline_input(
        f"\nPaste the {model.upper()} response for batch {batch_id}:"
    )

    # Parse
    parsed = parse_response(raw_response, expected_words)

    # Count flags
    flags = [r for r in parsed if r["flag"]]
    print(f"\nParsed {len(parsed)} lines. Flags: {len(flags)}")

    if flags:
        print(f"\n{len(flags)} flagged response(s) — this batch needs re-running.")
        print("Record this partial result anyway? (will be overwritten on re-run)")
        save_partial = input("Save partial result? (y/n): ").strip().lower()
        if save_partial == "y":
            record_results(parsed, model, records, run_number)
            save_results(records)
            print("Partial results saved.")
        print("\nPlease re-run this batch in a FRESH session.")
        return

    # All clean — record and save
    record_results(parsed, model, records, run_number)
    save_results(records)
    print(f"\nAll results clean. Saved to {RESULTS_FILE}.")
    print(f"Batch {batch_id} complete for {model}.")


if __name__ == "__main__":
    main()
