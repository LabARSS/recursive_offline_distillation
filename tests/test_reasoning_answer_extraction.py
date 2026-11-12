"""Thinking-token answers stay within the first assistant turn."""

from types import SimpleNamespace

import pytest

from core.datasets.gpqa import GPQAReasoningResponseDataset
from core.datasets.gsm8k.gsm8k_reasoning_response_dataset import GSM8KReasoningResponseDataset
from core.datasets.hendrycks_math import HendrycksMathReasoningResponseDataset
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import InvalidAnswerError, QADatasetConfig
from core.utils.cot_answers import CHAT_END_TOKENS


@pytest.fixture(
    params=[
        (GSM8KReasoningResponseDataset, "18", "99"),
        (HendrycksMathReasoningResponseDataset, "18", "99"),
        (MMLUReasoningResponseDataset, "a", "b"),
        (GPQAReasoningResponseDataset, "a", "b"),
    ],
    ids=["gsm8k", "math", "mmlu", "gpqa"],
)
def dataset(request):
    dataset_cls, gold, wrong = request.param
    # Phi-4-mini does not list <|end|> in all_special_tokens.
    tokenizer = SimpleNamespace(
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        all_special_tokens=["<think>", "</think>", "<|endoftext|>", "<pad>"],
    )
    return dataset_cls(tokenizer, QADatasetConfig(path="unused", dataset_id="test")), gold, wrong


def test_gpqa_training_preserves_options_and_uses_the_teacher_answer():
    tokenizer = SimpleNamespace(thinking_start_token="<think>", thinking_end_token="</think>")
    ds = GPQAReasoningResponseDataset(tokenizer, QADatasetConfig(path="unused", dataset_id="gpqa"))
    row = {
        "question_id": "17",
        "question": " Choose the correct option. ",
        "options": "['First', 'Second', 'Third', 'Fourth']",
        "answer": "B",
        "distill_reasoning": " The teacher chose First. ",
        "distill_answer": " A ",
    }

    assert ds.user_prompt(row) == (
        "Question: Choose the correct option.\nOptions:\na. First\nb. Second\nc. Third\nd. Fourth\n"
    )
    assert "Choose a correct option letter" in ds.system_prompt(row)
    assert ds.row_id(row) == "17"
    assert ds.assistant_response(row) == "<think>The teacher chose First.</think>a"


@pytest.mark.parametrize("suffix", ["<|end|>", "\n<|end|><|endoftext|>", "<|end|>\n<|end|>"])
@pytest.mark.parametrize("with_thinking_end", [True, False])
def test_phi_answers_remove_unlisted_end_markers(dataset, suffix, with_thinking_end):
    ds, gold, _ = dataset
    prefix = "Need final only answer.</think>" if with_thinking_end else ""
    assert ds.verify_assistant_response({"answer": gold}, f"{prefix}{gold}{suffix}") == (gold, True)


@pytest.mark.parametrize("end_token", CHAT_END_TOKENS)
@pytest.mark.parametrize("with_thinking_end", [True, False])
@pytest.mark.parametrize("first_correct", [True, False])
def test_later_turn_cannot_replace_the_first_answer(dataset, end_token, with_thinking_end, first_correct):
    ds, gold, wrong = dataset
    first_answer = gold if first_correct else wrong
    prefix = "<think>Derivation.</think>" if with_thinking_end else ""
    response = f"{prefix}{first_answer}{end_token}<|assistant|><think>Revision.</think>{gold}<|endoftext|>"
    assert ds.verify_assistant_response({"answer": gold}, response) == (first_answer, first_correct)


def test_later_turn_cannot_supply_a_missing_final_answer(dataset):
    ds, gold, _ = dataset
    response = f"<think>Unfinished.</think><|end|><|assistant|><think>Revision.</think>{gold}"
    assert ds.verify_assistant_response({"answer": gold}, response) == ("", False)


def test_empty_bare_answer_keeps_the_invalid_answer_contract(dataset):
    ds, gold, _ = dataset
    with pytest.raises(InvalidAnswerError):
        ds.verify_assistant_response({"answer": gold}, "<|end|><|assistant|>Revision.</think>" + gold)
