from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.gpqa.gpqa_single_token_response_dataset import GPQASingleTokenResponseDataset, QADatasetConfig
from core.distillation.distill import DistillationConfig, distill_on_dataset

if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(
        DistillationConfig(
            out_filename=str(
                Path(__file__).parent.joinpath(
                    "../../../../../data/out/splits/random/gpqa/train_distilled_deepseek_v4_flash_extend_w_v4_pro_short.parquet"
                )
            ),
            model="qwen/qwen3.8-max-0902",
            dataset=GPQASingleTokenResponseDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(
                        Path(__file__).parent.joinpath(
                            "../../../../../data/out/splits/random/gpqa/train_distilled_deepseek_v4_flash_extend_w_v4_pro_short.parquet"
                        )
                    ),
                    dataset_id="gpqa",
                ),
            ),
            regenerate_incorrect=True,
        )
    )
