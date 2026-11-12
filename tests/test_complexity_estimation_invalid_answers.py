"""An empty decoded answer must retain enough diagnostics to identify its raw token."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter


class _ResponseModel:
    device = "cpu"

    def __init__(self, token_ids, vocab_size):
        self.token_ids = token_ids
        self.vocab_size = vocab_size

    def generate(self, input_ids, attention_mask, **kwargs):
        appended = torch.tensor([self.token_ids], dtype=input_ids.dtype)
        return SimpleNamespace(
            sequences=torch.cat([input_ids, appended], dim=1),
            scores=(torch.zeros(1, self.vocab_size),),
        )


@pytest.mark.parametrize(
    ("response_kind", "valid"),
    [
        ("thinking", False),
        ("eos", False),
        ("newline", False),
        ("space", False),
        ("number", True),
        ("multi_token_number", True),
    ],
)
def test_response_tokens_saved_before_answer_verification(
    thinking_tokenizer, tmp_path, capsys, response_kind, valid
):
    raw_response = {
        "thinking": thinking_tokenizer.thinking_start_token,
        "eos": thinking_tokenizer.eos_token,
        "newline": "\n",
        "space": " ",
        "number": "4",
        "multi_token_number": "42",
    }[response_kind]
    token_ids = thinking_tokenizer.encode(raw_response, add_special_tokens=False)
    if not valid:
        assert len(token_ids) == 1
    source_path = tmp_path / "source.parquet"
    pd.DataFrame([{"question_id": "q0", "question": "What is two plus two?", "answer": "4"}]).to_parquet(
        source_path, index=False
    )
    adapter = QADatasetAdapter(
        GSM8KDirectResponseDataset(
            tokenizer=thinking_tokenizer,
            config=QADatasetConfig(path=str(source_path), dataset_id="gsm8k"),
        )
    )
    out_path = tmp_path / "estimate.parquet"
    runner = ComplexityEstimationRunner(
        config=ComplexityEstimationRunnerConfig(
            out_path=str(out_path),
            answer_field_name="estimation_phase_answer",
            answer_correctness_field_name="estimation_phase_answer_correctness",
            generate_config=ModelGenerateConfig(max_new_tokens=len(token_ids)),
        ),
        complexity_estimator=SingleTokenEntropyWithRandomEstimator(),
    )

    runner.estimate(adapter, _ResponseModel(token_ids, len(thinking_tokenizer)))

    output = capsys.readouterr().out
    assert "Invalid answer for row" not in output
    assert f"Invalid answers: {int(not valid)}" in output
    result = pd.read_parquet(out_path)
    saved_tokens = result.loc[0, "estimation_phase_answer_token_ids"]
    assert saved_tokens.tolist() == token_ids
    assert thinking_tokenizer.decode(saved_tokens, skip_special_tokens=False) == raw_response
    measurement = result.loc[0, ["estimation_phase_answer", "entropy_value", "random_value"]]
    if valid:
        assert measurement.notna().all()
    else:
        assert measurement.isna().all()
