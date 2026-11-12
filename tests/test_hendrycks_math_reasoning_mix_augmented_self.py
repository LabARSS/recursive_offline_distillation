"""Extend the mix without dropping inherited chains or reusing row IDs."""

import pandas as pd
import pytest

from src.preprocessing.build_hendrycks_math_reasoning_mix_10k import Trace, build_mix
from src.preprocessing.build_hendrycks_math_reasoning_mix_10k_augmented_self import extend_mix


@pytest.fixture
def inputs():
    train = pd.DataFrame([{"question_id": "17", "question": "An original math problem.", "answer": "2", "answer_w_steps": "Reference."}])
    test = pd.DataFrame([{"question_id": "0", "question": "A held-out math problem.", "answer": "7"}])
    previous, _, _ = build_mix(train, test, [Trace(
        "existing", "source-id", train.iloc[0].question, "An inherited complete derivation.\n",
        "2", "distill_reasoning", True,
    )])
    omi = pd.DataFrame([
        {"question_id": "source-1", "question": train.iloc[0].question, "answer": "2", "answer_w_steps": "Reference.",
         "distill_reasoning": r"Another solution. \boxed{2}", "distill_ans_correct": True, "problem_source": "math"},
        {"question_id": "source-2", "question": "A new augmented math problem.", "answer": "3", "answer_w_steps": "Augmented reference.",
         "distill_reasoning": r"Complete augmented work. \boxed{3}", "distill_ans_correct": True, "problem_source": "augmented_math"},
    ])
    student = pd.DataFrame([{"question_id": "17", "question": train.iloc[0].question, "answer": "2",
        "distill_reasoning": "A correct student derivation.\n", "distill_answer": "2", "distill_ans_correct": True,
        "pass": "hendrycks_math_train_greedy"}])
    return train, test, previous, omi, {"qwen_3b": student}


def test_combines_all_sources_with_unique_row_ids_and_intact_inherited_data(inputs):
    snapshots = [frame.copy(deep=True) for frame in inputs[:4]]
    output, audit = extend_mix(*inputs)
    assert len(output) == 4 and output.question_id.is_unique and output.trace_id.is_unique
    assert output.source.value_counts().to_dict() == {
        "existing": 1, "omi2_math": 1, "omi2_augmented_math": 1, "self_qwen_3b": 1,
    }
    assert output[output.original_question_id == "17"].question_id.nunique() == 3
    assert output[output.source == "existing"].iloc[0].distill_reasoning == "An inherited complete derivation.\n"
    assert output[output.source == "self_qwen_3b"].iloc[0].distill_reasoning == "A correct student derivation.\n"
    assert audit.status.eq("included").all()
    for frame, snapshot in zip(inputs[:4], snapshots):
        pd.testing.assert_frame_equal(frame, snapshot)


def test_omi2_overlap_does_not_duplicate_inherited_chains(inputs):
    train, test, _, omi, students = inputs
    previous, _, _ = build_mix(train, test, [Trace(
        "omi2_math", "old-omi2", train.iloc[0].question, omi.iloc[0].distill_reasoning, "2", "distill_reasoning", True,
    )])
    output, audit = extend_mix(train, test, previous, omi, students)
    assert len(output) == 3
    assert audit.status.eq("duplicate_chain").sum() == 1
    assert set(previous.trace_id) <= set(output.trace_id)


def test_flat_omi2_whitespace_changes_do_not_create_extra_chains(inputs):
    train, test, _, omi, students = inputs
    inherited_reasoning = "\n " + omi.iloc[0].distill_reasoning + "\n"
    previous, _, _ = build_mix(train, test, [Trace(
        "omi2_math", "raw-shard", train.iloc[0].question, inherited_reasoning, "2", "generated_solution", True,
    )])
    output, audit = extend_mix(train, test, previous, omi, students)
    assert len(output) == 3
    assert audit.status.eq("duplicate_chain").sum() == 1
    assert output[output.trace_id == previous.iloc[0].trace_id].iloc[0].distill_reasoning == inherited_reasoning


@pytest.mark.parametrize("kind,status", [
    ("false_flag", "source_marked_incorrect"),
    ("wrong_actual_answer", "final_answer_incorrect"),
    ("too_long", "at_or_over_character_limit"),
    ("test_overlap", "test_overlap"),
])
def test_excludes_bad_augmented_examples_using_actual_generated_answers(inputs, kind, status):
    train, test, previous, omi, students = inputs
    if kind == "false_flag": omi.loc[1, "distill_ans_correct"] = False
    elif kind == "wrong_actual_answer": omi.loc[1, "distill_reasoning"] = r"Wrong result. \boxed{9}"
    elif kind == "too_long": omi.loc[1, "distill_reasoning"] = "x" * 10000 + r"\boxed{3}"
    elif kind == "test_overlap": omi.loc[1, "question"] = test.iloc[0].question
    output, audit = extend_mix(train, test, previous, omi, students)
    assert "omi2_augmented_math" not in set(output.source)
    assert audit[audit.source == "omi2_augmented_math"].status.tolist() == [status]


def test_self_distillation_requires_flag_and_independent_correct_answer(inputs):
    train, test, previous, omi, students = inputs
    student = students["qwen_3b"].iloc[0].to_dict()
    students["qwen_3b"] = pd.DataFrame([
        student,
        student | {"distill_reasoning": "Incorrect derivation.", "distill_answer": "9"},
        student | {"distill_reasoning": "Flagged incorrect derivation.", "distill_ans_correct": False},
        student | {"distill_reasoning": "x" * 10000},
        student | {"distill_reasoning": "A distinct sampled fallback.", "pass": "hendrycks_math_train_t0.7_s0"},
    ])
    output, audit = extend_mix(train, test, previous, omi, students)
    assert output.source.eq("self_qwen_3b").sum() == 2
    assert set(audit[audit.source == "self_qwen_3b"].status) == {
        "included", "final_answer_incorrect", "source_marked_incorrect", "at_or_over_character_limit",
    }


def test_cross_student_identical_chains_are_not_reweighted(inputs):
    train, test, previous, omi, students = inputs
    students["phi4_mini"] = students["qwen_3b"].copy()
    output, audit = extend_mix(train, test, previous, omi, students)
    assert len(output) == 4
    assert audit[audit.source == "self_phi4_mini"].status.tolist() == ["duplicate_chain"]
