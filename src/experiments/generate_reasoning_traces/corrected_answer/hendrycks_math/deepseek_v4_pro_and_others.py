"""Generate correction continuations for remaining incorrect MATH traces.

Run deepseek_v4_flash_regenerate_incorrect_w_large.py first.
Requires OPENROUTER_API_TOKEN. Continuations are saved as corrected_reasoning;
the original distill_reasoning is preserved, and distill_answer uses the
provided ground-truth answer, following the other correction scripts.

Usage:
  python src/experiments/generate_reasoning_traces/corrected_answer/hendrycks_math/deepseek_v4_pro_and_others.py
"""

from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.hendrycks_math import HendrycksMathCorrectedAnswerDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.distillation.distill import DistillationConfig, DistillationResultWriter, distill_on_dataset


class CorrectedAnswerResultWriter(DistillationResultWriter):
    def write_to_df(self, df, config, result):
        row = df.iloc[result.index].to_dict()
        df.at[result.index, config.field_ans] = config.dataset.assistant_response(row)
        df.at[result.index, config.field_reasoning] = result.answer
        df.at[result.index, config.field_ans_correct] = True


if __name__ == "__main__":
    freeze_support()

    repo_root = Path(__file__).resolve().parents[5]
    output_path = repo_root / "data/out/distillation/hendrycks_math_corrected_answer_deepseek_v4_pro.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distill_on_dataset(
        DistillationConfig(
            out_filename=str(output_path),
            model="~deepseek/deepseek-pro-latest",
            dataset=HendrycksMathCorrectedAnswerDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(
                        repo_root
                        / "data/out/distillation/hendrycks_math_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"
                    ),
                    dataset_id="hendrycks_math_distilled_deepseek_v4_flash_regenerate_incorrect_w_large",
                ),
            ),
            field_reasoning="corrected_reasoning",
            regenerate_incorrect=True,
        ),
        distillation_result_writer=CorrectedAnswerResultWriter(),
    )
