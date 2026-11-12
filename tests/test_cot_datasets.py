"""CPU-only contracts for prompted CoT answers and mathematical equivalence."""

from types import SimpleNamespace

import pytest

from core.datasets.gpqa import GPQACoTResponseDataset
from core.datasets.gsm8k import GSM8KCoTResponseDataset
from core.datasets.hendrycks_math import HendrycksMathCoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.utils import cot_answers


MATH_DATASETS = [GSM8KCoTResponseDataset, HendrycksMathCoTResponseDataset]


def _dataset(dataset_cls):
    tokenizer = SimpleNamespace(all_special_tokens=["<eos>", "<pad>", "<think>", "</think>"])
    return dataset_cls(tokenizer=tokenizer, config=QADatasetConfig(path="unused", dataset_id="test"))


@pytest.mark.parametrize("dataset_cls", MATH_DATASETS)
def test_math_prompt_preserves_question_and_repeats_final_answer_format(dataset_cls):
    dataset = _dataset(dataset_cls)
    row = {"question": " Solve this problem. ", "question_id": "17", "answer": "private_gold_answer"}
    assert dataset.user_prompt(row).startswith("Solve this problem.\n\n")
    assert "Answer: <answer>" in dataset.user_prompt(row)
    assert "step-by-step" in dataset.system_prompt(row)
    assert "Answer: <answer>" in dataset.system_prompt(row)
    assert row["answer"] not in dataset.system_prompt(row) + dataset.user_prompt(row)
    assert dataset.row_id(row) == "17"
    with pytest.raises(NotImplementedError, match="evaluation only"):
        dataset.assistant_response(row)


def test_gpqa_prompt_preserves_existing_option_order():
    dataset = _dataset(GPQACoTResponseDataset)
    row = {
        "question": " Choose. ",
        "options": "['First', 'Second', 'Third', 'Fourth']",
        "question_id": "17",
        "answer": "private_gold_answer",
    }
    assert dataset.user_prompt(row).startswith("Question: Choose.\nOptions:\na. First\nb. Second\nc. Third\nd. Fourth\n\n")
    assert "[[letter]]" in dataset.user_prompt(row)
    assert "step-by-step" in dataset.system_prompt(row)
    assert "[[letter]]" in dataset.system_prompt(row)
    assert row["answer"] not in dataset.system_prompt(row) + dataset.user_prompt(row)
    assert dataset.row_id(row) == "17"
    with pytest.raises(NotImplementedError, match="evaluation only"):
        dataset.assistant_response(row)


@pytest.mark.parametrize("dataset_cls", [*MATH_DATASETS, GPQACoTResponseDataset])
def test_adapter_uses_chat_template_without_thinking_tokens(dataset_cls):
    dataset = _dataset(dataset_cls)
    seen = []

    def apply_chat_template(messages, **kwargs):
        seen.append((messages, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    dataset.tokenizer.apply_chat_template = apply_chat_template
    row = {"question": "Question", "answer": "a", "question_id": "17", "options": "['First', 'Second']"}
    tokenized = QADatasetAdapter(dataset).process_row(row)
    assert tokenized.row_id == "17"
    assert tokenized.input_ids == [1, 2]
    assert tokenized.labels == []
    assert seen == [
        (
            [
                {"role": "system", "content": dataset.system_prompt(row)},
                {"role": "user", "content": dataset.user_prompt(row)},
            ],
            {"tokenize": True, "add_generation_prompt": True, "return_dict": True},
        )
    ]


@pytest.mark.parametrize("dataset_cls", MATH_DATASETS)
@pytest.mark.parametrize(
    ("response", "answer", "correct"),
    [
        ("Reasoning.\nAnswer: 2", "2", True),
        ("<think>Reasoning.</think>\nAnswer: 2<eos><pad>\n", "2", True),
        ("Reasoning.\nanswer: 2", "2", True),
        ("Answer: 3\nCorrection.\nAnswer: 2", "2", True),
        ("Answer: 2\nCorrection.\nAnswer: 3", "3", False),
        ("Answer: 2\nMore reasoning.", "", False),
        ("The result is 2.", "", False),
        ("Reasoning.\nAnswer:", "", False),
        ("Reasoning.\nAnswer:<eos>", "", False),
        ("Reasoning.\nAnswer: \\frac{1}{", r"\frac{1}{", False),
        ("", "", False),
    ],
)
def test_math_grades_only_the_final_marked_answer(dataset_cls, response, answer, correct):
    assert _dataset(dataset_cls).verify_assistant_response({"answer": "2"}, response) == (answer, correct)


@pytest.mark.parametrize(
    ("response", "answer", "correct"),
    [
        ("Reasoning.\n[[b]]", "b", True),
        ("Reasoning.\n[[ B ]]<eos><pad>", "b", True),
        ("The correct answer is: [[b]]<eos>", "b", True),
        ("b. Second[[b]]<eos>", "b", True),
        ("7. Therefore, choose Second. [[b]]<eos>", "b", True),
        ("[[a]]\nCorrection.\n[[b]]", "b", True),
        ("[[b]]\nCorrection.\n[[a]]", "a", False),
        ("[[b]]\nStep-by-step reasoning:\nMore reasoning.<eos>", "b", True),
        ("[[b]]\nCorrection. The answer is [[a]].", "a", False),
        ("[[b]]<|end|><|assistant|>Correction.\n[[a]]<|endoftext|>", "b", True),
        ("[[a]]<|end|><|assistant|>Correction.\n[[b]]<|endoftext|>", "a", False),
        ("[[b<|end|><|assistant|>Correction.\n[[b]]<|endoftext|>", "", False),
        ("[[b]]\nCorrection.\n[[a", "", False),
        ("[[b]]\nCorrection.\n[[bc]]", "", False),
        ("[[b]]\nCorrection.\n[[z]]", "", False),
        ("Answer: b", "", False),
        ("Therefore, the correct answer is b.", "", False),
        ("b", "", False),
        (r"\[\boxed{b}\]", "", False),
        ("[[b", "", False),
        ("[[bc]]", "", False),
        ("[[b.]]", "", False),
        ("[[z]]", "", False),
        ("[[]]", "", False),
        ("", "", False),
    ],
)
def test_gpqa_requires_a_valid_last_marked_option_letter(response, answer, correct):
    row = {"answer": "b", "options": "['First', 'Second', 'Third', 'Fourth']"}
    assert _dataset(GPQACoTResponseDataset).verify_assistant_response(row, response) == (answer, correct)


@pytest.mark.parametrize(
    ("gold", "prediction", "correct"),
    [
        ("0.5", r"\frac{1}{2}", True),
        ("0", r"\frac{0}{2}", True),
        ("0.333333", r"\frac{1}{3}", True),
        (r"\frac{x+2}{7}", r"\frac{2+x}{7}", True),
        ("x+x", "2x", True),
        (r"\frac{1}{\frac{2}{3}}", r"\frac{3}{2}", True),
        (r"\frac{A}{B}", r"\dfrac{A}{B}", True),
        (r"\frac{A}{B}", r"\frac{a}{b}", False),
        (r"\Delta", r"\delta", False),
        ("x", "y", False),
        (r"\{1,2\}", r"\{2,1\}", True),
        ("[0,1)", r"[0,\frac{2}{2})", True),
        ("[0,1)", "[0,1]", False),
        ("[0,1)", "(0,1)", False),
        ("(1,2)", r"\{1,2\}", False),
        ("(1,2)", "(2,1)", False),
        ("1,2,3", "1,2", False),
        ("4,6,14,15", "461415", False),
        ("1,000", "1000", True),
        ("0.5", r"\(\frac{1}{2}\)", True),
        ("0.5", "$$0.5$$", True),
        ("0.5", r"\boxed{\frac{1}{2}}", True),
        ("90\\text{ square\nunits}", r"90\text{ square units}", True),
        ("no solution", "no solution", True),
        ("2", "3", False),
        ("2", "not math", False),
        ("2", r"\badcommand{2}", False),
        ("2", "", False),
        ("", "", False),
    ],
)
def test_math_equivalence(gold, prediction, correct):
    assert cot_answers.verify_math_answer(gold, prediction) is correct


@pytest.mark.parametrize("gold", ["1,000", "1000"])
@pytest.mark.parametrize("prediction", ["1000", "1,000", "$1,000$", r"\boxed{1,000}"])
def test_gsm8k_normalizes_thousands_separators(gold, prediction):
    assert _dataset(GSM8KCoTResponseDataset).verify_assistant_response(
        {"answer": gold}, f"Reasoning.\nAnswer: {prediction}"
    ) == (prediction, True)


@pytest.mark.parametrize("stage", ["_parse_math_answer", "verify"])
@pytest.mark.parametrize("error", [ValueError("unsupported notation"), TimeoutError("timed out")])
def test_grading_failures_count_as_incorrect(monkeypatch, stage, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cot_answers, stage, fail)
    assert cot_answers.verify_math_answer("0.5", r"\frac{1}{2}") is False
