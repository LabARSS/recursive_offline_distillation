"""Exercise greedy trace exports without loading a model or calling a GPU."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.generate_reasoning_traces.reasoning_trace.hendrycks_math import (
    phi4_mini_self,
    qwen_3b_self,
    self_distillation_shared as shared,
)


@pytest.mark.parametrize("entrypoint,model_id,model_name", [
    (phi4_mini_self, "microsoft/Phi-4-mini-instruct", "phi4_mini"),
    (qwen_3b_self, "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
])
def test_one_greedy_pass_exports_correct_complete_traces_with_train_ids(
    tmp_path, monkeypatch, entrypoint, model_id, model_name,
):
    monkeypatch.setattr(shared, "REPO_ROOT", tmp_path)
    train_path = tmp_path / "data/source/hendrycks_math/hendrycks_math_train.parquet"
    train_path.parent.mkdir(parents=True)
    train = pd.DataFrame([
        {"question_id": qid, "question": f"Question {qid}", "answer": answer, "answer_w_steps": "Reference"}
        for qid, answer in [("12", "2"), ("31", "3"), ("40", "4"), ("50", "5"), ("62", "6")]
    ])
    train.to_parquet(train_path, index=False)
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>", all_special_tokens=["<eos>", "<pad>"])
    loaded_models, calls = [], []

    def fake_tokenizer(name, **kwargs):
        loaded_models.append(name)
        return tokenizer

    class FakeEvaluator:
        def __init__(self, config, tokenizer):
            self.config = config

        def evaluate(self):
            calls.append(self.config)
            output = Path(self.config.out_path) / shared.GREEDY_DATASET_ID
            output.mkdir(parents=True)
            pd.DataFrame([
                {"row_id": "12", "response": "First derivation.\nAnswer: 2<eos><pad>", "parsed_answer": "2", "is_correct": True, "is_truncated": False},
                {"row_id": "31", "response": "Wrong derivation.\nAnswer: 7", "parsed_answer": "7", "is_correct": False, "is_truncated": False},
                {"row_id": "40", "response": "Incomplete reasoning.\nAnswer: 4", "parsed_answer": "4", "is_correct": True, "is_truncated": True},
                {"row_id": "50", "response": "Answer: 5", "parsed_answer": "5", "is_correct": True, "is_truncated": False},
                {"row_id": "62", "response": "Second derivation.\nAnother step.\n\nAnswer: 6\n", "parsed_answer": "6", "is_correct": True, "is_truncated": False},
            ]).to_parquet(output / "responses.parquet", index=False)

    monkeypatch.setattr(shared.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(shared, "Evaluator", FakeEvaluator)
    entrypoint.main()

    assert loaded_models == [model_id]
    assert len(calls) == 1
    config = calls[0]
    assert config.generation.temperature == 0.0
    assert config.generation.max_new_tokens == 4096
    assert config.eval_dataset.dataset.dataset_id == shared.GREEDY_DATASET_ID
    assert not config.eval_dataset.add_thinking_start_token
    assert tokenizer.pad_token == tokenizer.eos_token
    output = tmp_path / f"data/out/distillation/hendrycks_math_self_distilled_{model_name}.parquet"
    result = pd.read_parquet(output)
    assert result.question_id.tolist() == ["12", "62"]
    assert result.distill_reasoning.tolist() == ["First derivation.", "Second derivation.\nAnother step."]
    assert result.distill_answer.tolist() == ["2", "6"]
    assert result.distill_ans_correct.all()
    assert result["pass"].tolist() == [shared.GREEDY_DATASET_ID] * 2
    assert result.question.tolist() == ["Question 12", "Question 62"]
