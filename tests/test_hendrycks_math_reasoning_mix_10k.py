"""Original question identity, intact chains, answer correctness, and leakage."""

import pandas as pd
import pytest

from src.preprocessing.build_hendrycks_math_reasoning_mix_10k import (
    Trace, build_mix, deepseek_traces, omi2_traces, qwen3_traces,
)


@pytest.fixture
def splits():
    train = pd.DataFrame([
        {"question_id": "17", "question": "Compute the training value.", "answer": "2", "answer_w_steps": "Reference."},
        {"question_id": "29", "question": "Another training value.", "answer": "3", "answer_w_steps": "Reference."},
    ])
    test = pd.DataFrame([{"question_id": "17", "question": "Compute the test value.", "answer": "2"}])
    return train, test


def trace(**changes):
    values = dict(source="example", source_record_id="foreign-999", question="Compute the training value.",
                  reasoning="An intact derivation.\n", prediction="2", reasoning_column="reasoning")
    return Trace(**(values | changes))


def test_different_sources_and_traces_share_original_train_id(splits):
    train, test = splits
    out, _, _ = build_mix(train, test, [trace(), trace(source="other", reasoning="A different derivation.")])
    assert out.question_id.tolist() == ["17", "17"]
    assert out.trace_id.is_unique
    assert set(out.distill_reasoning) == {"An intact derivation.\n", "A different derivation."}


def test_whitespace_duplicate_uses_first_train_id_and_original_text(splits):
    train, test = splits
    train.loc[2] = {"question_id": "88", "question": "Compute  the training value.", "answer": "2", "answer_w_steps": "Reference."}
    out, _, aliases = build_mix(train, test, [trace(question="Compute  the training value.")])
    assert out.iloc[0].question_id == "17"
    assert out.iloc[0].question == train.iloc[0].question
    assert aliases == [{"question_id": "88", "canonical_question_id": "17"}]


def test_length_limit_is_strict_and_never_cuts_a_trace(splits):
    train, test = splits
    kept = " " + "x" * 9997 + "\n"
    out, audit, _ = build_mix(train, test, [trace(reasoning=kept), trace(reasoning="y" * 10000)])
    assert out.distill_reasoning.tolist() == [kept]
    assert audit.status.tolist() == ["included", "at_or_over_character_limit"]


@pytest.mark.parametrize("candidate,status", [
    (trace(source_correct=False), "source_marked_incorrect"),
    (trace(source_correct=True, prediction="7"), "final_answer_incorrect"),
    (trace(prediction=""), "missing_final_answer"),
    (trace(reasoning=" \n"), "empty_reasoning"),
    (trace(question="Unknown training question."), "not_in_original_train"),
    (trace(reasoning="partial </think> text"), "embedded_control_marker"),
])
def test_rejects_incorrect_or_invalid_chains(splits, candidate, status):
    out, audit, _ = build_mix(*splits, [candidate])
    assert out.empty
    assert audit.status.tolist() == [status]


def test_source_train_label_or_id_cannot_admit_a_test_question(splits):
    train, test = splits
    out, audit, _ = build_mix(train, test, [trace(question=test.iloc[0].question, source_record_id="train-17")])
    assert out.empty and audit.status.tolist() == ["test_overlap"]


def test_shared_13_words_is_excluded_even_for_an_original_train_question(splits):
    train, test = splits
    shared = "one two three four five six seven eight nine ten eleven twelve thirteen"
    train.loc[0, "question"] = shared + " training suffix"
    test.loc[0, "question"] = shared + " testing suffix"
    out, audit, _ = build_mix(train, test, [trace(question=train.iloc[0].question)])
    assert out.empty and audit.status.tolist() == ["test_overlap"]


def test_math_symbol_case_is_preserved_when_matching_questions(splits):
    train, test = splits
    train.loc[0, "question"] = "Evaluate A/B."
    out, audit, _ = build_mix(train, test, [trace(question="Evaluate a/b.")])
    assert out.empty and audit.status.tolist() == ["not_in_original_train"]


def test_omi2_verifies_generated_answer_instead_of_expected_label(splits):
    frame = pd.DataFrame([{"problem_source": "math", "problem": "Compute the training value.",
                           "expected_answer": "2", "generated_solution": r"Wrong work. \boxed{7}"}])
    out, audit, _ = build_mix(*splits, omi2_traces(frame))
    assert out.empty and audit.status.tolist() == ["final_answer_incorrect"]


def test_extra_detailed_uses_entire_written_answer_as_reasoning(splits):
    written = "A complete written derivation.\n\nAnswer: 2"
    frame = pd.DataFrame([{"question": "Compute the training value.", "distill_reasoning": "DO NOT USE",
                           "distill_answer": written, "distill_ans_correct": True}])
    out, _, _ = build_mix(*splits, deepseek_traces(frame, extra_detailed=True))
    assert out.iloc[0].distill_reasoning == written
    assert out.iloc[0].distill_answer == "2"
    assert out.iloc[0].source_reasoning_column == "distill_answer"


def test_qwen_preserves_reasoning_payload_and_requires_thinking_end(splits):
    payload = "\nAn uncut Qwen derivation.\n"
    frame = pd.DataFrame([
        {"problem": "Compute the training value.", "conversations": [{"role": "assistant", "content": f"<think>{payload}</think>\\boxed{{2}}"}]},
        {"problem": "Compute the training value.", "conversations": [{"role": "assistant", "content": "<think>Unfinished"}]},
    ])
    out, audit, _ = build_mix(*splits, qwen3_traces(frame))
    assert out.distill_reasoning.tolist() == [payload]
    assert audit.status.tolist() == ["included", "empty_reasoning"]


def test_identical_chains_do_not_get_duplicate_weight(splits):
    out, audit, _ = build_mix(*splits, [trace(), trace(source="other", source_record_id="another")])
    assert len(out) == 1
    assert audit.status.tolist() == ["included", "duplicate_chain"]
