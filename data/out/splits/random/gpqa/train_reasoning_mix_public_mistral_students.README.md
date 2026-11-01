# GPQA public mix with Mistral and student reasoning

Dataset: `train_reasoning_mix_public_mistral_students.parquet`

This separate training dataset contains **2,011 traces covering all 437 local
training questions**. It retains every row and trace ID from the 1,459-row
`train_reasoning_mix_public.parquet` mixture, and adds the eligible Mistral and
base-student CoT traces below. The previous mixture and all source files are read
only during the build.

| Source | Retained traces |
| --- | ---: |
| Originally correct teacher traces | 396 |
| Original expert explanations | 436 |
| sxiong written solutions | 371 |
| spawn99 solutions | 256 |
| Mistral Large 4.0 | 187 |
| Llama 3B base-model CoT | 104 |
| Phi-4 Mini base-model CoT | 136 |
| Qwen 3B base-model CoT | 125 |
| **Total** | **2,011** |

Mistral input: `train_distilled_mistral_large_4_0.parquet`. Of its 437 rows, 350
have correct final answers. Of those, 187 fit the character budget and 163 exceed
it. The remaining 87 are source-graded incorrect and are excluded.

Student inputs are `train_self_distill_llama_3b_cot.parquet`,
`train_self_distill_phi4_mini_cot.parquet`, and
`train_self_distill_qwen_3b_cot.parquet`. These are the existing correct CoT
exports from the base models on `train_original`. Two Qwen rows (local question
IDs 44 and 385) contain only an answer letter, with no reasoning, and are excluded.

Every addition requires a true source correctness flag and a stored final answer
whose selected option text matches the local training reference. Correctness here
means final-answer agreement; it does not independently certify every scientific
claim in the reasoning. The source option order is preserved. Question membership
is checked by case-preserving Unicode NFC and whitespace normalization, rather
than trusting a filename, source split, or numeric ID.

Every full target fits **10,000 characters**, counting
`<think>reasoning</think>answer`. The longest is **9,976 characters**. No trace is
truncated. The 109 local test questions and their IDs have zero overlap with this
dataset. The 25 teacher traces with appended corrections remain excluded from the
inherited teacher source; other independent solutions for those training
questions are eligible.

Deduplication checks the same question, same option order, identical trimmed
reasoning text, and same final letter. Different chains for one question remain.
No exact duplicates were found among these additions.

The adjacent `.manifest.json` records source counts, input/output hashes, budget
details, and a snapshot of the previous mixture's manifest. The `.audit.csv`
records every candidate's inclusion/exclusion outcome. Matched source subsets are
saved under `train_reasoning_mix_public_mistral_students_sources/`.

Build from the repository root:

```bash
.venv/bin/python src/preprocessing/build_gpqa_mistral_student_reasoning_mix.py
.venv/bin/python -m pytest tests/test_gpqa_mistral_student_reasoning_mix.py tests/test_gpqa_public_reasoning_mix.py -q
```

The existing GPQA reasoning runner can load this dataset with
`train_dataset="train_reasoning_mix_public_mistral_students"`. New entrypoints for
all three students are in the `public_reasoning_mix_mistral_students` and
`public_reasoning_mix_mistral_students_data_mix` GPQA experiment groups. Both use
learning rate `1e-4`, save at epochs 2/5/10/15/20, and disable separate
self-distillation. The second group also adds direct-answer examples.
