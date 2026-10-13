"""Final-answer extraction and end-to-end GSM8K/MATH grading with stub tokenizers."""

from types import SimpleNamespace

import pytest

from core.datasets.gsm8k import GSM8KCoTResponseDataset
from core.datasets.hendrycks_math import HendrycksMathCoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.utils.cot_answers import CHAT_END_TOKENS, clean_assistant_response, extract_math_answer, final_response_line


SPECIAL_TOKENS = ["<think>", "</think>", "<eos>", "<pad>", "<|eot_id|>", "<|im_end|>"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("", ""),
        (" \n\t\r\n", ""),
        ("First\nSecond\n", "Second"),
        ("First\r\nSecond\r\n", "Second"),
        ("First\rSecond", "Second"),
        ("First\n \t\nSecond\n \n", "Second"),
        ("Answer: 42<eos><pad>", "Answer: 42"),
        ("<think>Derivation</think>\nAnswer: 42<|eot_id|>", "Answer: 42"),
        ("Answer: 42\n<|im_end|>\n<pad>\n", "Answer: 42"),
        (r"Answer: \frac{A}{B}", r"Answer: \frac{A}{B}"),
    ],
)
def test_final_response_line(response, expected):
    assert final_response_line(response, SPECIAL_TOKENS) == expected


def test_overlapping_duplicate_and_empty_special_tokens_are_safe():
    # Longest tokens must be removed first; otherwise <end> leaves a stray 'tail'.
    tokens = ["", "<end>", "<end>tail", "<end>"]
    assert final_response_line("Answer: 42<end>tail", tokens) == "Answer: 42"


def test_special_tokens_may_be_an_iterator():
    assert final_response_line("Answer: 42<eos><pad>", iter(SPECIAL_TOKENS)) == "Answer: 42"


@pytest.mark.parametrize(
    "response",
    [
        "Reasoning.\nAnswer: 20<|end|><|endoftext|>",
        "Reasoning.\nAnswer: 20\n<|end|><|endoftext|>",
        "Reasoning.\nAnswer: 20<|end|>\n<|end|>",
    ],
)
def test_phi_end_marker_is_removed_when_tokenizer_does_not_list_it(response):
    special_tokens = ["<|endoftext|>"]
    assert final_response_line(response, iter(special_tokens)) == "Answer: 20"
    assert extract_math_answer(response, special_tokens) == "20"


@pytest.mark.parametrize("end_token", CHAT_END_TOKENS)
def test_cleanup_keeps_first_turn_before_removing_special_tokens(end_token):
    response = f"<think>Reasoning.</think>\nAnswer: 20{end_token}<|assistant|>Revision.\nAnswer: 99"
    assert clean_assistant_response(response, iter(SPECIAL_TOKENS)) == "Reasoning.\nAnswer: 20"
    assert extract_math_answer(response, SPECIAL_TOKENS) == "20"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Answer:42", "42"),
        ("Answer: 42", "42"),
        ("  ANSWER:\t42  ", "42"),
        ("answer: -0.75", "-0.75"),
        (r"Answer: \frac{A}{B}", r"\frac{A}{B}"),
        (r"Answer: \{1,2,3\}", r"\{1,2,3\}"),
        ("Answer: [0,1)", "[0,1)"),
        (r"Answer: \boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}"),
        ("Answer: 42<eos>\n<pad>", "42"),
        ("Answer: 1\nRevision.\nAnswer: 42", "42"),
        ("Answer: 42\nRevision.\nAnswer: 1", "1"),
        ("The answer is 42.", ""),
        ("Reasoning. Answer: 42", ""),
        ("Final Answer: 42", ""),
        ("**Answer:** 42", ""),
        ("Answer = 42", ""),
        ("Answer: 42\nMore reasoning.", ""),
        ("Answer: 42\nAnswer:", ""),
        ("Answer: \t<eos>", ""),
        ("Answer:\n42", ""),
        ("Answer: \\frac{1}{\n2}", ""),
        ("<eos><pad>", ""),
    ],
)
def test_extracts_only_a_marked_answer_on_the_final_line(response, expected):
    assert extract_math_answer(response, SPECIAL_TOKENS) == expected


@pytest.fixture(params=[GSM8KCoTResponseDataset, HendrycksMathCoTResponseDataset], ids=["gsm8k", "math"])
def dataset(request):
    return request.param(
        tokenizer=SimpleNamespace(all_special_tokens=SPECIAL_TOKENS),
        config=QADatasetConfig(path="unused", dataset_id="test"),
    )


@pytest.mark.parametrize(
    ("gold", "prediction", "correct"),
    [
        ("0.5", r"\frac{1}{2}", True),
        ("0.5", r"\frac{2}{3}", False),
        ("-0.75", r"-\frac{3}{4}", True),
        ("-0.75", r"\frac{3}{4}", False),
        ("0", r"\frac{0}{2}", True),
        ("42", r"\boxed{42.0}", True),
        ("42", r"\badcommand{42}", False),
    ],
)
def test_dataset_preserves_the_extracted_answer_and_uses_math_equivalence(dataset, gold, prediction, correct):
    row = {"answer": gold, "question_id": "17"}
    response = f"<think>Reasoning yields {gold}.</think>\nAnswer: {prediction}<eos><pad>"
    assert dataset.verify_assistant_response(row, response) == (prediction, correct)


def test_dataset_grades_phi_answer_with_an_unlisted_end_marker(dataset):
    dataset.tokenizer.all_special_tokens = ["<|endoftext|>"]
    assert dataset.verify_assistant_response(
        {"answer": "20"}, "Reasoning.\nAnswer: 20<|end|><|endoftext|>"
    ) == ("20", True)


@pytest.mark.parametrize(("first_answer", "correct"), [("20", True), ("99", False)])
def test_dataset_ignores_answers_from_generated_followup_turns(dataset, first_answer, correct):
    response = f"Answer: {first_answer}<|end|><|assistant|>Revision.\nAnswer: 20<|endoftext|>"
    assert dataset.verify_assistant_response({"answer": "20"}, response) == (first_answer, correct)


@pytest.mark.parametrize(
    "response",
    [
        "The answer is 42.",
        "Correct intermediate value: 42.\nAnswer: 41",
        "Answer: 42\nMore reasoning.",
        "Answer: 42\nAnswer:",
        "Answer: 42\nAnswer: \\frac{1}{",
        "Answer: 42\nAnswer: \\unknown{42}",
        "Answer: 42\nAnswer: 0/0",
    ],
)
def test_dataset_never_uses_a_correct_intermediate_answer(dataset, response):
    assert dataset.verify_assistant_response({"answer": "42"}, response)[1] is False


@pytest.mark.parametrize("gold", ["", " "])
def test_dataset_does_not_accept_an_empty_gold_answer(dataset, gold):
    assert dataset.verify_assistant_response({"answer": gold}, "Answer: 42") == ("42", False)


def test_math_dataset_preserves_multiline_gold_latex():
    dataset = HendrycksMathCoTResponseDataset(
        tokenizer=SimpleNamespace(all_special_tokens=[]),
        config=QADatasetConfig(path="unused", dataset_id="math"),
    )
    row = {"answer": "90\\text{ square\nunits}"}
    assert dataset.verify_assistant_response(row, r"Answer: 90\text{ square units}") == (
        r"90\text{ square units}",
        True,
    )


@pytest.mark.parametrize(
    ("gold", "prediction", "correct"),
    [
        ("x+x", "2x", True),
        ("X", "x", False),
        ("A+a", "a+A", True),
        ("A+a", "2a", False),
        (r"\{1,2\}", r"\{2,1\}", True),
        ("[0,1)", "[0,1]", False),
        ("(1,2)", r"\{1,2\}", False),
    ],
)
def test_math_dataset_grades_symbolic_and_structured_answers(gold, prediction, correct):
    dataset = HendrycksMathCoTResponseDataset(
        tokenizer=SimpleNamespace(all_special_tokens=SPECIAL_TOKENS),
        config=QADatasetConfig(path="unused", dataset_id="math"),
    )
    assert dataset.verify_assistant_response({"answer": gold}, f"Answer: {prediction}") == (prediction, correct)
