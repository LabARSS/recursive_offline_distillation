"""Train once, then evaluate the same GPQA checkpoints at both reasoning budgets."""

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.datasets.gpqa import GPQAReasoningResponseDataset
from experiments.distillation_on_synthetic_traces.gpqa import shared


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_name", ["qwen_3b", "llama_3b", "phi4_mini"])
def test_entrypoint_trains_once_and_evaluates_both_budgets(monkeypatch, tmp_path, model_name):
    events = []
    training_configs = []
    eval_configs = []
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")

    def load_tokenizer(path, **kwargs):
        assert path == str(REPO_ROOT / "artifacts/base_models_v0" / model_name)
        events.append("tokenizer")
        return tokenizer

    def setup_tokens(loaded_tokenizer):
        assert loaded_tokenizer is tokenizer
        tokenizer.thinking_start_token = "<think>"
        tokenizer.thinking_end_token = "</think>"
        events.append("thinking_tokens")

    class FakeTrainer:
        def __init__(self, config, tokenizer):
            training_configs.append(config)

        def train(self):
            events.append("train")

        def unload(self):
            events.append("unload")

    class FakeEvaluator:
        def __init__(self, config, tokenizer):
            eval_configs.append(config)

        def evaluate_all(self):
            events.append(f"evaluate_{eval_configs[-1].generation.max_thinking_tokens}")

    monkeypatch.setattr(shared.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(shared, "setup_thinking_tokens", setup_tokens)
    monkeypatch.setattr(shared, "LoRATrainer", FakeTrainer)
    monkeypatch.setattr(shared, "MultiCheckpointEvaluator", FakeEvaluator)
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module(
        f"experiments.distillation_on_synthetic_traces.gpqa.direct_reasoning_trace.{model_name}_head_truncated20000"
    )
    assert not events
    runpy.run_path(module.__file__, run_name="__main__")

    assert events == ["tokenizer", "thinking_tokens", "train", "unload", "evaluate_4096", "evaluate_8192"]
    assert len(training_configs) == 1
    train_config = training_configs[0]
    assert train_config.out_path == str(
        REPO_ROOT / "artifacts/distillation_on_synthetic_traces/gpqa/direct_reasoning_trace"
        / f"{model_name}_head_truncated20000"
    )
    assert isinstance(train_config.train_dataset.dataset, GPQAReasoningResponseDataset)
    assert train_config.train_dataset.dataset.config.path == str(
        REPO_ROOT / "data/out/splits/random/gpqa/train_distilled_deepseek_v4_flash_extend_w_v4_pro_head20000.parquet"
    )
    assert train_config.lora_training_args.train_thinking_token_embeddings
    if model_name == "phi4_mini":
        assert train_config.lora_training_args.target_modules == shared.phi4_mini_lora_target_modules

    assert len(eval_configs) == 2
    for config, budget in zip(eval_configs, [4096, 8192], strict=True):
        assert config.checkpoints_dir == train_config.out_path
        assert isinstance(config.eval_dataset.dataset, GPQAReasoningResponseDataset)
        assert config.eval_dataset.dataset.tokenizer is tokenizer
        assert config.eval_dataset.add_thinking_start_token
        assert config.eval_dataset.dataset.config.path == str(REPO_ROOT / "data/out/splits/random/gpqa/test.parquet")
        assert config.eval_dataset.dataset.dataset_id == f"gpqa_random_test_cap{budget}"
        assert config.generation.max_thinking_tokens == budget
        assert config.generation.max_new_tokens == budget + 10
        assert config.summary_filename == f"summary_reasoning_evals_cap{budget}.json"
