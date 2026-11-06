# Hendrycks MATH entropy-gain proportional distillation

These experiments apply the MMLU `entropy_gain_proportional` method to the
`hendrycks_math_reasoning_mix_under10k_omi2_aug_self` trace pool. They use the
detailed reasoning prompt and large LoRA adapters from the corresponding
`distillation_on_synthetic_traces/hendrycks_math/reasoning_mix_under10k_omi2_aug_self_detailed_prompt`
baselines for Llama 3B, Qwen 3B, and Phi-4-mini.

Each epoch estimates the student's normalized first-token entropy with the
direct-answer prompt, subtracts the mean normalized entropy of Llama 70B and
Qwen 72B, and samples proportionally to positive entropy gain. Fresh random
draws are recorded by `SingleTokenEntropyWithRandomEstimator` each epoch.
The reasoning sample is 10% of the training pool (rounded down), with ground-truth
direct answers added at approximately one quarter of that count. The direct-answer
count is adjusted by at most 32 rows so the combined sample fills effective batches
of 64 documents for packed training. The current 54,805-row pool therefore uses up
to 5,480 reasoning traces and 1,368 direct answers per epoch. Both sets are selected
by the same policy and shuffled before packing.
Direct answers retain MATH's LaTeX answer format.

Training uses seed 42, learning rate `1e-5`, the cosine schedule, LoRA rank 64
and alpha 128, trained thinking-token embeddings, and the existing per-model
packing budgets. Checkpoints are saved at epochs 10, 20, 50, 80, 100, 150, and
200. Evaluation uses the same detailed prompt as training, the MATH test set,
and a 4,096-token thinking cap.

Run the two proxy estimates first from the repository root. They cover the
exact training pool, including augmented questions, and write directly to
`data/out/single_token_entropy_normalized/<train_dataset>_<proxy>.parquet`.
The entropy estimator already normalizes by `log(vocab_size)`; these outputs
do not need the legacy normalization notebook.

```bash
PYTHONPATH=src uv run python -m experiments.estimate_single_token_entropy.hendrycks_math.llama_70b
PYTHONPATH=src uv run python -m experiments.estimate_single_token_entropy.hendrycks_math.qwen_72b
```

Then run any of the student experiments:

```bash
PYTHONPATH=src uv run python -m experiments.distillation_by_metrics.hendrycks_math.entropy_gain_proportional.llama_3b_lora_large_cap4096_shuffle
PYTHONPATH=src uv run python -m experiments.distillation_by_metrics.hendrycks_math.entropy_gain_proportional.qwen_3b_lora_large_cap4096_shuffle
PYTHONPATH=src uv run python -m experiments.distillation_by_metrics.hendrycks_math.entropy_gain_proportional.phi4_mini_lora_large_cap4096_shuffle
```

Student base models are loaded from `artifacts/base_models_v0/<model>`.
Outputs are saved under
`artifacts/distillation_by_metrics/hendrycks_math/entropy_gain_proportional/<model>_lora_large_shuffle/`,
including the merged teacher entropy, per-epoch resampling data, checkpoints,
and `summary_reasoning_evals_cap4096.json`.
