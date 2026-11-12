# GPQA public mix with Mistral and student traces: experiments

All six entrypoints use
`data/out/splits/random/gpqa/train_reasoning_mix_public_mistral_students.parquet`:
2,011 reasoning traces covering the 437 training questions.

| Experiment group | Reasoning examples | Direct-answer examples |
| --- | ---: | ---: |
| `public_reasoning_mix_mistral_students` | 2,011 | 0 |
| `public_reasoning_mix_mistral_students_data_mix` | 2,011 | 2,011 |

Each group has `llama_3b.py`, `phi4_mini.py`, and `qwen_3b.py`. Both groups
explicitly set `self_distillation=False`: the training dataset already includes
the eligible base-model reasoning traces from all three students.

Every entrypoint uses **learning rate `1e-4`** and saves checkpoints at epochs
**2, 5, 10, 15, and 20**. The shared GPQA runner supplies rank-64 LoRA,
model-specific packing and Phi-4 Mini target modules, and evaluation on the
unchanged 109-question test split with a 4,096-token thinking budget.

The direct-answer variant uses one single-letter example per dataset row,
preserving that row's option order. Its 4,022 training examples are merged before
packing. The reasoning-only variant has 2,011 examples.

Run from the repository root, for example:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.distillation_on_synthetic_traces.gpqa.public_reasoning_mix_mistral_students.llama_3b
PYTHONPATH=src .venv/bin/python -m experiments.distillation_on_synthetic_traces.gpqa.public_reasoning_mix_mistral_students_data_mix.llama_3b
```

Replace `llama_3b` with `phi4_mini` or `qwen_3b` for the other students.
Outputs go to separate directories under
`artifacts/distillation_on_synthetic_traces/gpqa/<experiment_group>/<model>/`.
