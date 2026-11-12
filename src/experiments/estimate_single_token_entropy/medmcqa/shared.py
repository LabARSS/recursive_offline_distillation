from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.datasets.gpqa.gpqa_single_token_response_dataset import GPQASingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.utils.device import DEVICE_MAP


def run(model_id: str, model_name: str, dataset: str = "medmcqa_train_10k"):
    repo_root = Path(__file__).resolve().parents[4]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map=DEVICE_MAP, torch_dtype="auto"
    )

    # SingleTokenEntropyEstimator normalizes at measurement time, so write straight to the
    # normalized directory read by distillation_by_metrics/medmcqa.
    ComplexityEstimationRunner(
        config=ComplexityEstimationRunnerConfig(
            out_path=(
                repo_root / "data/out/single_token_entropy_normalized" / f"{dataset}_{model_name}.parquet"
            ).as_posix(),
            answer_field_name="model_answer",
            answer_correctness_field_name="model_answer_correct",
            generate_config=ModelGenerateConfig(max_new_tokens=1),
            save_every=100,
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
    ).estimate(
        QADatasetAdapter(
            dataset=GPQASingleTokenResponseDataset(
                config=QADatasetConfig(
                    path=(repo_root / "data/source/medmcqa" / f"{dataset}.parquet").as_posix(),
                    dataset_id="medmcqa_proxy_entropy",
                ),
                tokenizer=tokenizer,
            )
        ),
        model,
    )
