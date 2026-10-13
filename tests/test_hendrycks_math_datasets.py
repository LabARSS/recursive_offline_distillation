"""Hendrycks MATH answer formatting and verification contracts."""

from types import SimpleNamespace

import pytest

from core.datasets.hendrycks_math import (
    HendrycksMathCorrectedAnswerDataset,
    HendrycksMathDirectResponseDataset,
    HendrycksMathReasoningResponseDataset,
)
from core.datasets.qa_dataset import InvalidAnswerError, QADatasetConfig


def _dataset(dataset_cls=HendrycksMathDirectResponseDataset):
    tokenizer = SimpleNamespace(
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        all_special_tokens=["<think>", "</think>", "<eos>", "<pad>"],
    )
    return dataset_cls(
        tokenizer=tokenizer,
        config=QADatasetConfig(path="unused", dataset_id="hendrycks_math"),
    )


@pytest.mark.parametrize("answer", ["42", r"\frac{A}{\sqrt{B}}", "(3, -1)", r"\{1, 2\}"])
def test_direct_answers_preserve_latex_and_symbols(answer):
    ds = _dataset()
    row = {"answer": f" {answer} "}
    assert ds.assistant_response(row) == answer
    assert ds.verify_assistant_response(row, f"\n{answer}\n") == (answer, True)


def test_direct_verification_preserves_variable_case():
    ds = _dataset()
    assert ds.verify_assistant_response({"answer": r"\frac{A}{B}"}, r"\frac{a}{b}") == (
        r"\frac{a}{b}",
        False,
    )


@pytest.mark.parametrize("response", ["", " \n "])
def test_direct_empty_answers_raise_invalid_answer_error(response):
    with pytest.raises(InvalidAnswerError):
        _dataset().verify_assistant_response({"answer": "0"}, response)


def test_reasoning_target_uses_distilled_fields_and_thinking_tokens():
    ds = _dataset(HendrycksMathReasoningResponseDataset)
    row = {
        "answer": "0",
        "distill_reasoning": " This derivation produces a fraction. ",
        "distill_answer": r" \frac{A}{B} ",
    }
    assert ds.assistant_response(row) == r"<think>This derivation produces a fraction.</think>\frac{A}{B}"


@pytest.mark.parametrize(
    ("response", "expected", "correct"),
    [
        (r"<think>Derivation.</think>\frac{A}{B}", r"\frac{A}{B}", True),
        (r"<think>Derivation.</think>\frac{A}{B}<eos><pad>", r"\frac{A}{B}", True),
        (r" \frac{A}{B} ", r"\frac{A}{B}", True),
        (r"<think>Derivation.</think>\frac{a}{b}", r"\frac{a}{b}", False),
        ("<think>Unfinished.</think><eos>", "", False),
    ],
)
def test_reasoning_verification_extracts_final_answer(response, expected, correct):
    ds = _dataset(HendrycksMathReasoningResponseDataset)
    assert ds.verify_assistant_response({"answer": r"\frac{A}{B}"}, response) == (expected, correct)


def test_correction_prompt_preserves_incorrect_and_correct_answers():
    ds = _dataset(HendrycksMathCorrectedAnswerDataset)
    row = {
        "question": " Simplify the expression. ",
        "answer": r"\frac{A}{B}",
        "distill_reasoning": " A partial derivation. ",
        "distill_answer": r"\frac{B}{A}",
    }
    prompt = ds.user_prompt(row)
    assert "Question: Simplify the expression." in prompt
    assert "Partial reasoning so far:\nA partial derivation." in prompt
    assert r"Incorrect answer this reasoning led to: \frac{B}{A}" in prompt
    assert r"Correct answer: \frac{A}{B}" in prompt


@pytest.mark.parametrize(
    ("response", "expected", "correct"),
    [
        (r"\frac{A}{B}", r"\frac{A}{B}", True),
        (r"Revised derivation. Answer: \frac{A}{B}", r"\frac{A}{B}", True),
        (r"Answer: 1. Revised derivation. Answer: \frac{A}{B}", r"\frac{A}{B}", True),
        (r"Revised derivation. Answer: \frac{a}{b}", r"\frac{a}{b}", False),
    ],
)
def test_correction_verification_uses_last_answer_marker(response, expected, correct):
    ds = _dataset(HendrycksMathCorrectedAnswerDataset)
    assert ds.verify_assistant_response({"answer": r"\frac{A}{B}"}, response) == (expected, correct)


def test_correction_empty_answer_raises_invalid_answer_error():
    ds = _dataset(HendrycksMathCorrectedAnswerDataset)
    with pytest.raises(InvalidAnswerError):
        ds.verify_assistant_response({"answer": "0"}, "Revised derivation. Answer: ")
