"""Generate DeepSeek V3 CoT traces for the 10k MedMCQA train sample.

Non-thinking CoT arrives in distill_answer. It is moved to distill_reasoning (without the final
[[letter]]) and distill_answer is set to the extracted letter, the GPQAReasoningResponseDataset
format. All rows go to medmcqa_distilled_cot_deepseek_v3.parquet, correct rows also to
medmcqa_distilled_cot_deepseek_v3_correct.parquet.

Run from the repository root:
  PYTHONPATH=src python -m experiments.generate_reasoning_traces.reasoning_trace.medmcqa.deepseek_v3
"""

from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd

from core.datasets.gpqa import GPQACoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.distillation.distill import DistillationConfig, distill_on_dataset


def to_reasoning_format(df: pd.DataFrame, dataset: GPQACoTResponseDataset) -> pd.DataFrame:
    """Move raw CoT from distill_answer to distill_reasoning; leave already-converted rows alone.

    The chain is cut at its first [[...]] marker; some traces keep reasoning (or loop) after
    answering. The answer and its correctness come from that first marker.
    """
    df = df.copy()
    raw = df["distill_answer"].fillna("").str.contains("[[", regex=False) & (
        df["distill_reasoning"].fillna("") == ""
    )
    for index, row in df[raw].iterrows():
        cot = row["distill_answer"]
        start = cot.find("[[")
        end = cot.find("]]", start)
        first_marker_response = cot[: end + 2] if end != -1 else cot
        answer, correct = dataset.verify_assistant_response(row.to_dict(), first_marker_response)
        df.at[index, "distill_reasoning"] = cot[:start].strip()
        df.at[index, "distill_answer"] = answer
        df.at[index, "distill_ans_correct"] = correct
    return df


if __name__ == "__main__":
    freeze_support()

    repo_root = Path(__file__).resolve().parents[5]
    output_path = repo_root / "data/out/distillation/medmcqa_distilled_cot_deepseek_v3.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = GPQACoTResponseDataset(
        tokenizer=None,  # type: ignore[reportArgumentType]
        config=QADatasetConfig(
            path=str(repo_root / "data/source/medmcqa/medmcqa_train_10k.parquet"),
            dataset_id="medmcqa_train_10k",
        ),
    )
    df = distill_on_dataset(
        DistillationConfig(
            out_filename=str(output_path),
            model="deepseek/deepseek-chat-v3.1",
            dataset=dataset,
        ),
        reasoning=False,
    )

    df = to_reasoning_format(df, dataset)
    df.to_parquet(output_path, index=False)

    correct = df[df["distill_ans_correct"].astype(bool)]
    correct_path = output_path.with_name(f"{output_path.stem}_correct.parquet")
    correct.to_parquet(correct_path, index=False)
    print(f"Wrote {len(correct)}/{len(df)} correct traces to {correct_path}")
