"""
2_prompt_builder.py
===================
Reads the generated test set CSV and produces batch prompt files —
one set for each language — ready to be copy-pasted into ChatGPT and Gemini.

Each batch contains 20 words. The prompt template is identical for every
batch, with only the modulus K and the word list changing.

Output files (in data/prompts/):
  mod3_batch001.txt ... mod3_batch050.txt
  mod4_batch001.txt ... mod4_batch050.txt
  mod5_batch001.txt ... mod5_batch050.txt

A master index file (data/prompts/index.csv) lists every batch file
with its language and word range for tracking purposes.
"""

import csv
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/test_set.csv"
PROMPTS_DIR  = "data/prompts"
INDEX_FILE   = os.path.join(PROMPTS_DIR, "index.csv")

os.makedirs(PROMPTS_DIR, exist_ok=True)

# ── Prompt template ───────────────────────────────────────────────────────────
# {K}      — replaced with the modulus (3, 4, or 5)
# {WORDS}  — replaced with the numbered word list
PROMPT_TEMPLATE = """\
You are given a formal language L over the alphabet {{a, b, c}} defined as:
L = {{ w | the number of occurrences of 'b' in w is divisible by {K} }}

For each word below, determine whether it belongs to L.

Rules:
- Do not execute any code or use any tools
- Do not explain your reasoning
- Answer with exactly one word per line: YES or NO
- Your response must contain exactly 20 lines in this format:
  N. [word] : YES
  or
  N. [word] : NO

{WORDS}"""


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TEST SET
# ══════════════════════════════════════════════════════════════════════════════

def load_test_set(filepath: str) -> dict:
    """
    Load the test set CSV and group records by batch_id.

    Returns:
        dict mapping batch_id -> list of record dicts, preserving order.
    """
    batches = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row["batch_id"]
            if bid not in batches:
                batches[bid] = []
            batches[bid].append(row)
    return batches


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PROMPT FOR ONE BATCH
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(batch_records: list, mod: int) -> str:
    """
    Build the full prompt string for a single batch.

    Args:
        batch_records : list of record dicts for this batch (20 words)
        mod           : modulus integer (3, 4, or 5)

    Returns:
        Formatted prompt string ready to copy-paste.
    """
    word_lines = []
    for i, record in enumerate(batch_records, start=1):
        word_lines.append(f"{i}. {record['word']}")
    words_block = "\n".join(word_lines)

    return PROMPT_TEMPLATE.format(K=mod, WORDS=words_block)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading test set from {INPUT_FILE}...")
    batches = load_test_set(INPUT_FILE)
    print(f"Loaded {len(batches)} batches.")

    index_rows = []

    for batch_id, records in sorted(batches.items()):
        # Extract modulus from batch_id (e.g. "mod3_batch001" -> 3)
        mod = int(batch_id.split("_")[0].replace("mod", ""))

        prompt      = build_prompt(records, mod)
        output_path = os.path.join(PROMPTS_DIR, f"{batch_id}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        index_rows.append({
            "batch_id":   batch_id,
            "language":   f"mod_{mod}",
            "num_words":  len(records),
            "first_word": records[0]["word"],
            "last_word":  records[-1]["word"],
            "file":       output_path,
        })

    # Write index
    with open(INDEX_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["batch_id", "language", "num_words",
                           "first_word", "last_word", "file"]
        )
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\nPrompt files written to {PROMPTS_DIR}/")
    print(f"Index written to {INDEX_FILE}")
    print(f"Total batch files: {len(index_rows)}")
    print("\nNext step: open each .txt file, copy its contents into")
    print("ChatGPT (or Gemini), and save the response for parsing.")


if __name__ == "__main__":
    main()
