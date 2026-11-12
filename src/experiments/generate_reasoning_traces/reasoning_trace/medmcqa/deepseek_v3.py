"""Generate DeepSeek V3 CoT traces for the 10k MedMCQA train sample.

Writes the raw CoT (in distill_answer) to medmcqa_distilled_cot_deepseek_v3.parquet, then the
correct rows in GPQAReasoningResponseDataset format (reasoning without the final [[letter]],
letter in distill_answer) to medmcqa_distilled_cot_deepseek_v3_correct.parquet.

Run from the repository root:
  PYTHONPATH=src python -m experiments.generate_reasoning_traces.reasoning_trace.medmcqa.deepseek_v3
"""

from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.gpqa import GPQACoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.distillation.distill import DistillationConfig, distill_on_dataset

if __name__ == "__main__":
    freeze_support()

    repo_root = Path(__file__).resolve().parents[5]
    output_path = repo_root / "data/out/distillation/medmcqa_distilled_cot_deepseek_v3.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = distill_on_dataset(
        DistillationConfig(
            out_filename=str(output_path),
            model="deepseek/deepseek-chat-v3.1",
            dataset=GPQACoTResponseDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(repo_root / "data/source/medmcqa/medmcqa_train_10k.parquet"),
                    dataset_id="medmcqa_train_10k",
                ),
            ),
        ),
        reasoning=False,
    )

    correct = df[df["distill_ans_correct"].astype(bool)].copy()
    # The answer goes after </think> in GPQAReasoningResponseDataset, so drop the final [[letter]].
    correct["distill_reasoning"] = correct["distill_answer"].map(lambda r: r[: r.rfind("[[")].strip())
    correct["distill_answer"] = correct["answer"].str.lower()
    correct_path = output_path.with_name(f"{output_path.stem}_correct.parquet")
    correct.to_parquet(correct_path, index=False)
    print(f"Wrote {len(correct)}/{len(df)} correct traces to {correct_path}")
