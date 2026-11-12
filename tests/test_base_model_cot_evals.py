"""Entrypoint-to-evaluator integration without downloading or running models."""

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.datasets.gpqa import GPQACoTResponseDataset
from core.datasets.gsm8k import GSM8KCoTResponseDataset
from core.datasets.hendrycks_math import HendrycksMathCoTResponseDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("gsm8k", GSM8KCoTResponseDataset, "data/source/gsm8k/gsm8k_test.parquet", "gsm8k_test"),
    ("gpqa", GPQACoTResponseDataset, "data/out/splits/random/gpqa/test.parquet", "gpqa_random_test"),
    (
        "hendrycks_math",
        HendrycksMathCoTResponseDataset,
        "data/source/hendrycks_math/hendrycks_math_test.parquet",
        "hendrycks_math_test",
    ),
]
MODELS = [
    ("qwen_3b_cot", "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
    ("phi4mini_cot", "microsoft/Phi-4-mini-instruct", "phi4_mini"),
    ("llama_3b_cot", "meta-llama/Llama-3.2-3B-Instruct", "llama_3b"),
]


@pytest.mark.parametrize(("dataset_name", "dataset_cls", "data_path", "dataset_id"), DATASETS)
@pytest.mark.parametrize(("entrypoint", "model_id", "model_name"), MODELS)
def test_entrypoint_builds_the_expected_eval_from_any_cwd(
    monkeypatch, tmp_path, dataset_name, dataset_cls, data_path, dataset_id, entrypoint, model_id, model_name
):
    module_name = f"experiments.distillation_on_synthetic_traces.{dataset_name}.base_model"
    shared = importlib.import_module(f"{module_name}.shared")
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
    tokenizers_loaded = []
    evaluations = []

    def load_tokenizer(path, **kwargs):
        tokenizers_loaded.append((path, kwargs))
        return tokenizer

    class FakeEvaluator:
        def __init__(self, config, tokenizer):
            self.config = config
            self.tokenizer = tokenizer

        def evaluate(self):
            evaluations.append((self.config, self.tokenizer))
            return [SimpleNamespace(accuracy=0.75, correct=3, total=4, num_truncated=1)]

    monkeypatch.setattr(shared.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(shared, "Evaluator", FakeEvaluator)
    monkeypatch.chdir(tmp_path)

    script = importlib.import_module(f"{module_name}.{entrypoint}")
    assert not tokenizers_loaded
    assert not evaluations
    runpy.run_path(script.__file__, run_name="__main__")

    assert tokenizers_loaded == [(model_id, {"trust_remote_code": True})]
    assert len(evaluations) == 1
    config, eval_tokenizer = evaluations[0]
    assert eval_tokenizer is tokenizer
    assert tokenizer.pad_token == "<eos>"
    assert config.model_path == model_id
    assert isinstance(config.eval_dataset.dataset, dataset_cls)
    assert config.eval_dataset.dataset.tokenizer is tokenizer
    assert config.eval_dataset.dataset.config.path == str(REPO_ROOT / data_path)
    assert config.eval_dataset.dataset.dataset_id == dataset_id
    assert config.eval_dataset.add_thinking_start_token is False
    assert config.out_path == str(
        REPO_ROOT / f"artifacts/distillation_on_synthetic_traces/{dataset_name}/base_model/{model_name}_cot"
    )
    assert config.generation.max_new_tokens == 8192
    assert config.generation.max_batch_size == 256
    assert config.generation.max_thinking_tokens is None
    assert config.generation.temperature == 0.0
    assert config.generation.top_p == 1.0
