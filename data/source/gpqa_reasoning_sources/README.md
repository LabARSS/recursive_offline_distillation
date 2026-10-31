# Public GPQA solution sources

These are separate source downloads. They include questions from both sides of our
local split and must not be used directly as training datasets. Use the matched
training-only mixture documented below.

| File | Rows | Contents |
| --- | ---: | --- |
| `original_explanations.parquet` | 546 | Original question-writer explanations and correct answer text |
| `sxiong_written_solutions.parquet` | 1,243 | Written model solutions, source labels, and metadata; no internal reasoning field |
| `spawn99_solutions.parquet` | 198 | Original nested DeepSeek/Claude responses, reasoning, answers, and grading metadata |

The official `Idavidrein/gpqa` CSV requires authenticated access. Its explanations
were obtained from the public `LLMcompe-Team-Watanabe/gpqa_preprocess` extended
release, which retains the original `Explanation` and correct answer text inside
its formatted answer. The downloaded file is retained under
`original/gpqa_extended_public.parquet`. All 546 question texts and correct answer
texts aligned with our local original dataset after whitespace normalization.

sxiong's `train.jsonl` and `test.jsonl` were processed as remote streams. Only the
written `content` and selected metadata were persisted. `reasoning_content` was
never written to a download file, temporary file, or cache. Source split names are
retained for provenance and do not define our local train/test membership.

Pinned repository revisions, retained columns, and SHA256 hashes are recorded in
`download_manifest.json`.

## Training-only mixture

The final mixture is:

`data/out/splits/random/gpqa/train_reasoning_mix_public.parquet`

It contains **1,484 distinct training traces covering all 437 training questions**:

| Source | Traces |
| --- | ---: |
| Current corrected traces | 421 |
| Original expert explanations | 436 |
| sxiong written solutions | 371 |
| spawn99 solutions | 256 |

The sxiong subset contains 299 DeepSeek-R1 written solutions and 72 source-graded
correct written responses from other models. Both the source correctness flag and
the actual final boxed letter must agree with the reference. Sixteen matched
training candidates with inconsistent/missing final letters were rejected, along
with eight reference-answer disagreements and 21 changed option sets.

The spawn99 subset contains 31 DeepSeek internal reasoning chains, 75 DeepSeek
written solutions, 62 standalone Claude responses, and 88 reasoning-assisted
Claude responses. Only source-graded correct solutions that fit the response
budget were included. The same question can have several different targets.

Every selected target has at most **10,000 characters**, counting
`<think>reasoning</think>answer`. The longest selected response is 9,976
characters. No solution was truncated. This filtering excluded one original
explanation, 20 sxiong written responses, and 44 spawn99 candidates that otherwise
passed the earlier eligibility checks.

Every row matches an exact local training question after Unicode NFC and whitespace
normalization. Case is preserved, including in mathematical symbols. All 109
held-out questions are excluded before checking source correctness or budget.
Neither external split labels nor external numeric IDs are trusted for matching.

External options are checked against the local option multiset. The source option
order is retained when a written solution refers to letters; its final target
letter is checked by the corresponding option text against our original correct
answer. This prevents shuffled options from silently changing the training label.

`question_id` is a unique trace identifier. `original_question_id` points to the
unchanged ID in `train_original.parquet`; `original_answer` retains that question's
original answer letter. Each row also records `source`, `source_variant`,
`source_model`, `source_record_id`, `source_split`, `option_order_changed`, and
`answer_verification`. Source grading and final-answer alignment do not establish
the correctness of every intermediate reasoning step.

Training-only source subsets are saved separately in:

`data/out/splits/random/gpqa/public_reasoning_sources/train_*.parquet`

The mixture has adjacent `.manifest.json` and `.audit.csv` files containing hashes,
source counts, budget details, and every candidate's inclusion/exclusion outcome.

## Reproduce

From the repository root:

```bash
.venv/bin/python src/preprocessing/download_gpqa_public_traces.py
.venv/bin/python src/preprocessing/build_gpqa_public_reasoning_mix.py --max-chars 10000
.venv/bin/python -m pytest tests/test_gpqa_public_reasoning_mix.py -q
```

The builder uses seed 42. Original train/test splits and current corrected traces
are read only. The mixed artifact can be used by the existing GPQA training runner
with `train_dataset="train_reasoning_mix_public"`.
