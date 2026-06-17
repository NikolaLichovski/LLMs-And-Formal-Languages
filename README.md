# Formal Language LLM Recognition Experiment
# README — Formal Languages and Finite Automata

## Languages Tested
- L1: { w ∈ {a,b,c}* | #b(w) ≡ 0 (mod 3) }
- L2: { w ∈ {a,b,c}* | #b(w) ≡ 0 (mod 4) }
- L3: { w ∈ {a,b,c}* | #b(w) ≡ 0 (mod 5) }

## Project Structure
```
1_generator.py       — generates test_set.csv (3000 words)
2_prompt_builder.py  — builds batch prompt .txt files
3_parser.py          — records model responses into results.csv
4_analysis.py        — computes all accuracy tables

data/
  test_set.csv       — generated test set (3000 words, all metadata)
  results.csv        — filled in after querying models
  prompts/           — 150 batch prompt .txt files (50 per language)
  analysis/          — output tables from analysis script
```

## Reproducibility
All randomness uses RANDOM_SEED = 42 (set in 1_generator.py).
Running 1_generator.py with this seed always produces the identical test set.

## Step-by-Step Workflow

### Step 1 — Generate test set
```
python 1_generator.py
```
Produces: data/test_set.csv

### Step 2 — Build batch prompts
```
python 2_prompt_builder.py
```
Produces: data/prompts/mod3_batch001.txt ... mod5_batch050.txt
          data/prompts/index.csv

### Step 3 — Query models and record responses
```
python 3_parser.py
```
Run once per batch per model. The script is interactive:
1. Select model (chatgpt or gemini)
2. Enter batch ID (e.g. mod3_batch001)
3. Copy the matching .txt file from data/prompts/ into the model's chat
4. Paste the model's response when prompted
5. Script parses, flags errors, saves to data/results.csv

Re-run policy:
- If any line is flagged (MISMATCH / MALFORMED / INVALID / MISSING),
  re-run the entire batch in a FRESH session
- Maximum 3 re-run attempts per batch per model
- After 3 failed attempts, flagged words are marked UNRESOLVABLE
  and excluded from accuracy calculations

Recommended order: complete all 50 batches for mod3 in ChatGPT,
then mod4, then mod5. Then repeat all 150 batches for Gemini.

### Step 4 — Run analysis
```
python 4_analysis.py
```
Produces all accuracy tables in data/analysis/.
Import these CSVs into Excel or similar to build charts.

## Batch Prompt Format
Each prompt defines the language, lists 20 words, and demands:
- No code execution or tool use
- No reasoning shown
- Exactly 20 lines of output in format: N. [word] : YES/NO

## CSV Columns

### test_set.csv
| Column              | Description                                      |
|---------------------|--------------------------------------------------|
| word                | The test word over {a,b,c}                       |
| language            | mod_3 / mod_4 / mod_5                            |
| length              | Word length (1–100)                              |
| b_count             | Number of b's in the word                        |
| b_count_tier        | very_low/low/medium/high/very_high               |
| distribution_pattern| clustered / interleaved / random                 |
| residue_class       | b_count mod k                                    |
| category            | positive / near_miss / negative                  |
| expected_answer     | YES or NO (ground truth)                         |
| batch_id            | e.g. mod3_batch001                               |

### results.csv (after querying)
All columns from test_set.csv, plus:
| Column              | Description                                      |
|---------------------|--------------------------------------------------|
| chatgpt_answer      | YES / NO / MISMATCH / MALFORMED / UNRESOLVABLE   |
| chatgpt_correct     | TRUE / FALSE / N/A                               |
| gemini_answer       | same as above                                    |
| gemini_correct      | same as above                                    |
| chatgpt_runs        | number of attempts needed for this batch         |
| gemini_runs         | same for gemini                                  |

## Analysis Tables Produced
1.  summary.csv                           — overall accuracy per model
2.  accuracy_by_language.csv              — mod3 vs mod4 vs mod5
3.  accuracy_by_length_range.csv          — 1-15, 16-30, 31-60, 61-100
4.  accuracy_by_category.csv             — positive vs near_miss vs negative
5.  accuracy_by_b_count_tier.csv         — very_low through very_high
6.  accuracy_by_distribution.csv         — clustered vs interleaved vs random
7.  accuracy_by_language_x_length.csv    — cross tabulation
8.  accuracy_by_language_x_category.csv  — cross tabulation
9.  accuracy_by_model_x_language_x_length.csv — three-way cross tabulation
10. consensus_errors.csv                 — words both models got wrong
11. model_specific_errors.csv            — words only one model got wrong
