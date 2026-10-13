"""Shared math grading across direct, thinking-token, and correction prompts."""

from types import SimpleNamespace

import pytest

from core.datasets.gsm8k.gsm8k_corrected_answer_dataset import GSM8KCorrectedAnswerDataset
from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset
from core.datasets.gsm8k.gsm8k_reasoning_response_dataset import GSM8KReasoningResponseDataset
from core.datasets.hendrycks_math import (
    HendrycksMathCorrectedAnswerDataset,
    HendrycksMathDirectResponseDataset,
    HendrycksMathReasoningResponseDataset,
)
from core.datasets.qa_dataset import InvalidAnswerError, QADatasetConfig


DATASET_CLASSES = {
    "gsm8k": {
        "direct": GSM8KDirectResponseDataset,
        "reasoning": GSM8KReasoningResponseDataset,
        "corrected": GSM8KCorrectedAnswerDataset,
    },
    "hendrycks_math": {
        "direct": HendrycksMathDirectResponseDataset,
        "reasoning": HendrycksMathReasoningResponseDataset,
        "corrected": HendrycksMathCorrectedAnswerDataset,
    },
}


def _dataset(family, mode, *, without_tokenizer=False):
    tokenizer = None if without_tokenizer else SimpleNamespace(
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        all_special_tokens=["<think>", "</think>", "<eos>", "<pad>"],
    )
    return DATASET_CLASSES[family][mode](
        tokenizer=tokenizer,
        config=QADatasetConfig(path="unused", dataset_id=family),
    )


def _response(mode, answer):
    if mode == "reasoning":
        return f"<think>A derivation.</think> {answer} <eos><pad>"
    if mode == "corrected":
        return f"A revised derivation.\nAnswer: {answer} "
    return f" \n{answer}\n "


@pytest.fixture(params=["direct", "reasoning", "corrected"])
def response_mode(request):
    return request.param


@pytest.fixture(params=["gsm8k", "hendrycks_math"])
def dataset(request, response_mode):
    return _dataset(request.param, response_mode)


@pytest.mark.parametrize(
    ("gold", "prediction", "correct"),
    [
        ("42", "42.0", True),
        ("0.5", r"\frac{2}{4}", True),
        ("-0.75", r"-\frac{3}{4}", True),
        ("1250", r"1.25\times 10^3", True),
        ("7", "$7$", True),
        ("42", r"\boxed{42}", True),
        ("0.25", r"25\%", True),
        ("25", r"25\%", False),
        ("1", "1.0000004", True),
        ("1", "1.0000006", False),
        ("42", "43", False),
        ("42", r"\unknowncommand{42}", False),
        ("42", "0/0", False),
    ],
)
def test_all_prompt_formats_grade_math_equivalence(dataset, response_mode, gold, prediction, correct):
    assert dataset.verify_assistant_response({"answer": gold}, _response(response_mode, prediction)) == (
        prediction,
        correct,
    )


@pytest.mark.parametrize(
    ("gold", "prediction", "correct"),
    [
        ("A+A", "2A", True),
        (r"\frac{A}{B}", "A/B", True),
        ("A+a", "a+A", True),
        ("X", "x", False),
        (r"\frac{A}{B}", r"\frac{a}{b}", False),
        (r"\{1,2\}", r"\{2,1\}", True),
        ("4,6,14,15", "15,14,6,4", True),
        ("(1,2)", r"\{1,2\}", False),
        ("[0,1)", "[0,1]", False),
        ("90\\text{ square\nunits}", r"90\text{ square units}", True),
        ("no solution", "no solution", True),
    ],
)
def test_math_prompts_preserve_case_and_structured_answers(response_mode, gold, prediction, correct):
    dataset = _dataset("hendrycks_math", response_mode)
    assert dataset.verify_assistant_response({"answer": gold}, _response(response_mode, prediction)) == (
        prediction,
        correct,
    )


@pytest.mark.parametrize(
    ("gold", "prediction", "correct"),
    [(1234, "1,234", True), ("-1234.5", "-1,234.5", True), (1234, "12,34", False)],
)
def test_gsm8k_prompts_handle_numeric_gold_and_thousands(response_mode, gold, prediction, correct):
    dataset = _dataset("gsm8k", response_mode)
    assert dataset.verify_assistant_response({"answer": gold}, _response(response_mode, prediction)) == (
        prediction,
        correct,
    )


@pytest.mark.parametrize("family", DATASET_CLASSES)
@pytest.mark.parametrize("mode", ["direct", "corrected"])
def test_trace_generation_grading_does_not_require_a_tokenizer(family, mode):
    dataset = _dataset(family, mode, without_tokenizer=True)
    assert dataset.verify_assistant_response({"answer": "0.5"}, _response(mode, r"\frac{1}{2}")) == (
        r"\frac{1}{2}",
        True,
    )


@pytest.mark.parametrize("family", DATASET_CLASSES)
@pytest.mark.parametrize("mode", ["direct", "corrected"])
@pytest.mark.parametrize("empty", ["", " \n\t"])
def test_empty_answers_keep_the_invalid_answer_contract(family, mode, empty):
    dataset = _dataset(family, mode)
    with pytest.raises(InvalidAnswerError):
        dataset.verify_assistant_response({"answer": "0"}, _response(mode, empty))


@pytest.mark.parametrize("family", DATASET_CLASSES)
def test_reasoning_with_empty_final_answer_is_incorrect(family):
    dataset = _dataset(family, "reasoning")
    assert dataset.verify_assistant_response({"answer": "0"}, _response("reasoning", "")) == ("", False)


@pytest.mark.parametrize("family", DATASET_CLASSES)
@pytest.mark.parametrize("mode", ["reasoning", "corrected"])
def test_bare_answer_fallback_uses_the_math_grader(family, mode):
    dataset = _dataset(family, mode)
    assert dataset.verify_assistant_response({"answer": "0.5"}, r" \frac{1}{2} ") == (
        r"\frac{1}{2}",
        True,
    )


@pytest.mark.parametrize("family", DATASET_CLASSES)
@pytest.mark.parametrize("final_answer", [r"\frac{1}{2}", r"\frac{2}{3}"])
def test_corrections_grade_only_the_last_answer_marker(family, final_answer):
    dataset = _dataset(family, "corrected")
    response = f"Answer: 0.5\nRevising the derivation. Answer: {final_answer}"
    assert dataset.verify_assistant_response({"answer": "0.5"}, response) == (
        final_answer,
        final_answer == r"\frac{1}{2}",
    )


@pytest.mark.parametrize("family", DATASET_CLASSES)
def test_correct_intermediate_reasoning_cannot_override_a_wrong_final_answer(family):
    dataset = _dataset(family, "reasoning")
    response = "<think>Intermediate result: 0.5.</think>0.6<eos>"
    assert dataset.verify_assistant_response({"answer": "0.5"}, response) == ("0.6", False)


def test_gsm8k_keeps_existing_lowercase_answer_storage():
    dataset = _dataset("gsm8k", "direct")
    assert dataset.verify_assistant_response({"answer": 1000}, r"1 \TIMES 10^3") == (r"1 \times 10^3", True)
