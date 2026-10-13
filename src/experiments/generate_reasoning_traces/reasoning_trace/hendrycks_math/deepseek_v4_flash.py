"""Generate reasoning traces for the Hendrycks MATH training split.

Requires OPENROUTER_API_TOKEN.

Usage:
  python src/experiments/generate_reasoning_traces/reasoning_trace/hendrycks_math/deepseek_v4_flash.py
"""

from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.hendrycks_math import HendrycksMathDirectResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.distillation.distill import DistillationConfig, distill_on_dataset

if __name__ == "__main__":
    freeze_support()

    repo_root = Path(__file__).resolve().parents[5]
    output_path = repo_root / "data/out/distillation/hendrycks_math_distilled_deepseek_v4_flash.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distill_on_dataset(
        DistillationConfig(
            out_filename=str(output_path),
            model="~deepseek/deepseek-flash-latest",
            dataset=HendrycksMathDirectResponseDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(repo_root / "data/source/hendrycks_math/hendrycks_math_train.parquet"),
                    dataset_id="hendrycks_math",
                ),
            ),
        )
    )
