"""
1_generator.py
==============
Generates the full test set of 3000 words (1000 per language) for the formal
language recognition experiment.

Languages tested:
  L1: { w in {a,b,c}* | #b(w) ≡ 0 (mod 3) }
  L2: { w in {a,b,c}* | #b(w) ≡ 0 (mod 4) }
  L3: { w in {a,b,c}* | #b(w) ≡ 0 (mod 5) }

Controlled variables per word:
  - language      : mod 3, mod 4, or mod 5
  - length        : sampled from defined distribution
  - residue_class : determines category (positive / near-miss / negative)
  - b_count_tier  : very_low / low / medium / high / very_high
  - distribution  : clustered / interleaved / random

Output:
  data/test_set.csv
"""

import csv
import random
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Output path ───────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data/test_set.csv"

# ── Language definitions ──────────────────────────────────────────────────────
LANGUAGES = [3, 4, 5]          # moduli
ALPHABET   = ["a", "b", "c"]  # full alphabet; only 'b' is counted

# ── Test set composition per language ─────────────────────────────────────────
# Total: 1000 words per language
# Categories:
#   positive  : 400  (#b ≡ 0 mod k)
#   near_miss : 200  (#b ≡ 1 mod k)
#   negative  : 400  (all other residue classes, evenly distributed)
CATEGORY_COUNTS = {
    "positive":  400,
    "near_miss": 200,
    "negative":  400,
}

# Residue classes assigned to negatives (excludes 0 = positive, 1 = near-miss)
NEGATIVE_RESIDUES = {
    3: [2],
    4: [2, 3],
    5: [2, 3, 4],
}

# ── Length distribution ───────────────────────────────────────────────────────
# Applied proportionally across all three categories.
# Ranges are inclusive on both ends.
LENGTH_RANGES = [
    (1,  15,  0.15),  # 15% → 60 pos/neg, 30 near-miss
    (16, 30,  0.35),  # 35% → 140 pos/neg, 70 near-miss
    (31, 60,  0.35),  # 35% → 140 pos/neg, 70 near-miss
    (61, 100, 0.15),  # 15% → 60 pos/neg, 30 near-miss
]

# ── B-count tiers ─────────────────────────────────────────────────────────────
# Applied to ALL words. Mod condition takes priority if a conflict arises.
# Tiers are distributed as evenly as possible within each category.
B_COUNT_TIERS = {
    "very_low":  (0,  5),
    "low":       (6,  15),
    "medium":    (16, 30),
    "high":      (31, 50),
    "very_high": (51, None),  # None means up to word length
}
TIER_ORDER = ["very_low", "low", "medium", "high", "very_high"]

# ── Distribution patterns ─────────────────────────────────────────────────────
# Applied in rotation within each length bucket for even representation.
# For words of length <= 5, only 'random' is used.
DISTRIBUTIONS = ["clustered", "interleaved", "random"]


# ══════════════════════════════════════════════════════════════════════════════
# MEMBERSHIP VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

def verify_membership(word: str, mod: int) -> bool:
    """
    Classical counting algorithm for membership verification.

    Scans the word left to right, counts occurrences of 'b',
    and checks whether that count is divisible by mod.

    Correctness: The language is defined purely by #b(w) mod k.
    This algorithm computes exactly that value with no approximation.

    Complexity: O(n) time, O(1) space — single left-to-right scan,
    one counter variable.

    Args:
        word : the input string over {a, b, c}
        mod  : the modulus k (3, 4, or 5)

    Returns:
        True if #b(word) ≡ 0 (mod k), False otherwise.
    """
    count = sum(1 for ch in word if ch == "b")
    return count % mod == 0


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: B-COUNT TIER LABEL
# ══════════════════════════════════════════════════════════════════════════════

def get_tier_label(b_count: int) -> str:
    """Return the b_count_tier label for a given absolute b count."""
    for tier, (lo, hi) in B_COUNT_TIERS.items():
        if hi is None:
            if b_count >= lo:
                return tier
        elif lo <= b_count <= hi:
            return tier
    return "very_low"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: FIND VALID B-COUNT
# ══════════════════════════════════════════════════════════════════════════════

def min_length_for_residue(residue: int, mod: int) -> int:
    """
    Return the minimum word length that can satisfy b_count ≡ residue (mod k).
    The smallest valid b_count is `residue` itself (when residue > 0),
    or 0 (when residue == 0). The word must be at least that long.
    """
    return residue if residue > 0 else 0


def find_b_count(length: int, residue: int, mod: int,
                 target_tier: str) -> int:
    """
    Find a valid b-count satisfying:
      1. b_count ≡ residue (mod k)              [mod condition — top priority]
      2. b_count falls within target_tier range  [secondary]
      3. 0 <= b_count <= length

    If no b-count in the target tier satisfies the mod condition,
    fall back to the nearest valid b-count within [0, length].

    NOTE: caller must ensure length >= min_length_for_residue(residue, mod).

    Args:
        length      : word length
        residue     : required residue class
        mod         : modulus
        target_tier : preferred b-count tier label

    Returns:
        A valid integer b-count, or -1 if impossible for this length.
    """
    lo, hi = B_COUNT_TIERS[target_tier]
    if hi is None:
        hi = length
    hi = min(hi, length)
    lo = max(lo, 0)

    # Collect all b-counts in [lo, hi] that satisfy the residue
    candidates = [b for b in range(lo, hi + 1) if b % mod == residue]

    if candidates:
        return random.choice(candidates)

    # Fallback: any b-count in [0, length] satisfying the residue
    fallback = [b for b in range(0, length + 1) if b % mod == residue]
    if fallback:
        return random.choice(fallback)

    # Truly impossible for this length (e.g. length=1, residue=2, mod=3)
    return -1


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: PLACE B'S ACCORDING TO DISTRIBUTION PATTERN
# ══════════════════════════════════════════════════════════════════════════════

def build_word(length: int, b_count: int, pattern: str) -> str:
    """
    Construct a word of given length with exactly b_count 'b' characters,
    distributed according to the specified pattern. Non-b positions are
    filled with 'a' or 'c' chosen uniformly at random.

    Patterns:
      clustered   — all b's are placed consecutively, block starts at a
                    random position
      interleaved — b's are spaced as evenly as possible across the word
      random      — b positions chosen uniformly at random

    Args:
        length  : total word length
        b_count : number of 'b' characters to place
        pattern : 'clustered', 'interleaved', or 'random'

    Returns:
        A string of length `length` over {a, b, c}.
    """
    # Edge cases
    if b_count == 0:
        return "".join(random.choice(["a", "c"]) for _ in range(length))
    if b_count == length:
        return "b" * length

    positions = set()

    if pattern == "clustered":
        # Place all b's in a contiguous block starting at a random index
        max_start = length - b_count
        start = random.randint(0, max_start)
        positions = set(range(start, start + b_count))

    elif pattern == "interleaved":
        # Space b's as evenly as possible using integer step
        step = length / b_count
        positions = set(int(i * step) for i in range(b_count))
        # Resolve collisions if any (rare floating point edge case)
        while len(positions) < b_count:
            positions.add(random.randint(0, length - 1))

    else:  # random
        positions = set(random.sample(range(length), b_count))

    # Build the word
    chars = []
    for i in range(length):
        if i in positions:
            chars.append("b")
        else:
            chars.append(random.choice(["a", "c"]))
    return "".join(chars)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: EXPAND LENGTH DISTRIBUTION INTO INDIVIDUAL LENGTHS
# ══════════════════════════════════════════════════════════════════════════════

def sample_lengths(total: int) -> list:
    """
    Sample `total` word lengths from the defined LENGTH_RANGES distribution.
    Each range contributes proportionally. Remainder words go to the largest
    range to avoid rounding loss.

    Returns a shuffled list of `total` integer lengths.
    """
    lengths = []
    counts = []
    for lo, hi, prop in LENGTH_RANGES:
        counts.append(round(total * prop))

    # Fix rounding so sum == total
    diff = total - sum(counts)
    counts[-1] += diff  # adjust last bucket

    for (lo, hi, _), count in zip(LENGTH_RANGES, counts):
        for _ in range(count):
            lengths.append(random.randint(lo, hi))

    random.shuffle(lengths)
    return lengths


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_category(mod: int, residue: int, count: int,
                      category: str, seen_words: set) -> list:
    """
    Generate `count` unique words for a given language (mod) and residue class.

    Distributes:
      - lengths according to LENGTH_RANGES
      - b_count_tiers evenly across TIER_ORDER (cycling)
      - distribution patterns evenly (cycling), random-only for length <= 5

    Deduplication:
      seen_words is a shared set across the entire language (all categories).
      When a generated word already exists in seen_words, up to MAX_RETRIES
      attempts are made with the same length. If all retries collide (space
      exhausted at that length), length is bumped by 1 and retried. This
      handles the hard ceiling at short lengths (e.g. length 1 only has 3
      possible words over {a,b,c}).

    Words whose sampled length is too short to satisfy the residue class
    have their length extended to the minimum required value.

    Args:
        mod        : modulus (3, 4, or 5)
        residue    : required residue class for b-count
        count      : number of words to generate
        category   : label string ('positive', 'near_miss', 'negative')
        seen_words : shared set of already-used words (mutated in place)

    Returns:
        List of dicts, each representing one test word with all metadata.
    """
    MAX_RETRIES = 20  # attempts at same length before bumping length

    lengths     = sample_lengths(count)
    min_len     = min_length_for_residue(residue, mod)
    records     = []
    tier_idx    = 0
    pattern_idx = 0

    for base_length in lengths:
        # Enforce minimum length for this residue class
        length = max(base_length, min_len)

        # Cycle through tiers and patterns for even distribution
        target_tier = TIER_ORDER[tier_idx % len(TIER_ORDER)]
        tier_idx   += 1
        pattern     = DISTRIBUTIONS[pattern_idx % len(DISTRIBUTIONS)] \
                      if length > 5 else "random"
        if length > 5:
            pattern_idx += 1

        # Try to generate a unique word; bump length if space is exhausted
        word    = None
        b_count = None
        attempt = 0

        while word is None or word in seen_words:
            if attempt > 0 and attempt % MAX_RETRIES == 0:
                # Exhausted attempts at this length — bump up by 1
                length += 1

            b_count = find_b_count(length, residue, mod, target_tier)

            if b_count == -1:
                length  = residue
                b_count = residue

            pat  = pattern if length > 5 else "random"
            word = build_word(length, b_count, pat)
            attempt += 1

        seen_words.add(word)
        actual_tier = get_tier_label(b_count)

        # Double-check with verifier
        membership = verify_membership(word, mod)
        expected   = "YES" if residue == 0 else "NO"

        # Sanity assertion — should never fire after fixes above
        assert membership == (residue == 0), (
            f"Verifier mismatch: word={word}, mod={mod}, "
            f"b_count={b_count}, residue={residue}, membership={membership}"
        )

        records.append({
            "word":                 word,
            "language":             f"mod_{mod}",
            "length":               length,
            "b_count":              b_count,
            "b_count_tier":         actual_tier,
            "distribution_pattern": pattern if length > 5 else "random",
            "residue_class":        b_count % mod,
            "category":             category,
            "expected_answer":      expected,
        })

    return records


def generate_language(mod: int) -> list:
    """
    Generate all 1000 test words for a single language (mod k).

    Composition:
      400 positive  (residue 0)
      200 near-miss (residue 1)
      400 negative  (remaining residues, evenly split)

    A single seen_words set is shared across all categories so that
    no word appears more than once in the entire language test set.

    Args:
        mod : modulus (3, 4, or 5)

    Returns:
        List of 1000 record dicts, all words unique within this language.
    """
    seen_words = set()  # shared across all categories for this language
    records    = []

    # Positive
    records += generate_category(mod, 0, 400, "positive", seen_words)

    # Near-miss
    records += generate_category(mod, 1, 200, "near_miss", seen_words)

    # Negative — evenly split across available residue classes
    neg_residues = NEGATIVE_RESIDUES[mod]
    base         = 400 // len(neg_residues)
    remainder    = 400 %  len(neg_residues)

    for idx, res in enumerate(neg_residues):
        count = base + (1 if idx < remainder else 0)
        records += generate_category(mod, res, count, "negative", seen_words)

    return records


def main():
    all_records = []

    for mod in LANGUAGES:
        print(f"Generating 1000 words for mod_{mod}...")
        lang_records = generate_language(mod)

        # Shuffle within each language (fixed seed already set globally)
        random.shuffle(lang_records)

        # Add batch assignment (batches of 20, per language)
        for i, record in enumerate(lang_records):
            record["batch_id"] = f"mod{mod}_batch{(i // 20) + 1:03d}"

        all_records += lang_records
        print(f"  Done. {len(lang_records)} words generated.")

    # Write CSV
    fieldnames = [
        "word", "language", "length", "b_count", "b_count_tier",
        "distribution_pattern", "residue_class", "category",
        "expected_answer", "batch_id",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nTest set written to {OUTPUT_FILE}")
    print(f"Total words: {len(all_records)}")

    # Quick summary
    print("\nSummary by language and category:")
    from collections import Counter
    counts = Counter((r["language"], r["category"]) for r in all_records)
    for (lang, cat), n in sorted(counts.items()):
        print(f"  {lang} | {cat:10s} : {n}")


if __name__ == "__main__":
    main()
