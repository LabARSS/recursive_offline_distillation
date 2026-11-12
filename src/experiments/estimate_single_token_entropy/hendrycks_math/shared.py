from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.datasets.hendrycks_math.hendrycks_math_direct_response_dataset import HendrycksMathDirectResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.utils.device import DEVICE_MAP


def run(model_id: str, model_name: str, train_dataset: str):
    repo_root = Path(__file__).resolve().parents[4]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map=DEVICE_MAP, torch_dtype="auto"
    )

    # SingleTokenEntropyEstimator normalizes at measurement time. Write directly to
    # the normalized directory so these estimates never need the legacy notebook.
    ComplexityEstimationRunner(
        config=ComplexityEstimationRunnerConfig(
            out_path=(
                repo_root / "data/out/single_token_entropy_normalized" / f"{train_dataset}_{model_name}.parquet"
            ).as_posix(),
            answer_field_name="model_answer",
            answer_correctness_field_name="model_answer_correct",
            generate_config=ModelGenerateConfig(max_new_tokens=1),
            save_every=100,
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
    ).estimate(
        QADatasetAdapter(
            dataset=HendrycksMathDirectResponseDataset(
                config=QADatasetConfig(
                    path=(repo_root / "data/out/distillation" / f"{train_dataset}.parquet").as_posix(),
                    dataset_id="hendrycks_math_proxy_entropy",
                ),
                tokenizer=tokenizer,
            )
        ),
        model,
    )
