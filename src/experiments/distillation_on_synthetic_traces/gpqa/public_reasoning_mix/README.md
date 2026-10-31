# GPQA public reasoning mixture experiments

All variants train on `data/out/splits/random/gpqa/train_reasoning_mix_public.parquet`.
There are entrypoints for `llama_3b`, `phi4_mini`, and `qwen_3b` in each group:

| Experiment group | Public reasoning mix | Direct-answer examples | Own correct CoT examples |
| --- | --- | --- | --- |
| `public_reasoning_mix` | Yes | No | No |
| `public_reasoning_mix_data_mix` | Yes | Yes | No |
| `public_reasoning_mix_self_distill` | Yes | No | Yes |
| `public_reasoning_mix_data_mix_self_distill` | Yes | Yes | Yes |

Every entrypoint explicitly uses **learning rate `1e-4`** and the checkpoint
schedule `[2, 5, 10, 15, 20]`. The existing shared GPQA runner supplies rank-64
LoRA, model-specific packing, and evaluation with a 4,096-token thinking budget
on the unchanged `test.parquet` split.

Direct-answer variants add one single-letter target for each row of the public
mixture, retaining that row's option order. Self-distillation variants additionally
load `train_self_distill_<model>_cot.parquet`, containing the original model's own
correct CoT responses on the training split.

Run from the repository root, for example:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.distillation_on_synthetic_traces.gpqa.public_reasoning_mix.llama_3b
PYTHONPATH=src .venv/bin/python -m experiments.distillation_on_synthetic_traces.gpqa.public_reasoning_mix_data_mix.llama_3b
PYTHONPATH=src .venv/bin/python -m experiments.distillation_on_synthetic_traces.gpqa.public_reasoning_mix_self_distill.llama_3b
PYTHONPATH=src .venv/bin/python -m experiments.distillation_on_synthetic_traces.gpqa.public_reasoning_mix_data_mix_self_distill.llama_3b
```

Replace `llama_3b` with `phi4_mini` or `qwen_3b` for the other models. Outputs are
isolated under `artifacts/distillation_on_synthetic_traces/gpqa/<group>/<model>/`.
