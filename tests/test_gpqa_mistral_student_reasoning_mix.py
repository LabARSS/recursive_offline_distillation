"""Checks for extending the GPQA mix without leaks or changed inherited rows."""

import pandas as pd
import pytest

from src.preprocessing.build_gpqa_mistral_student_reasoning_mix import (
    collect_local_traces, extend_mix,
)
from src.preprocessing.build_gpqa_public_reasoning_mix import Trace, build_mix, option_list


@pytest.fixture
def dataset():
    train = pd.DataFrame([{"question_id": "17", "question": "Training question?",
        "options": str(["right", "wrong 1", "wrong 2", "wrong 3"]), "answer": "A"}])
    test = pd.DataFrame([{"question_id": "42", "question": "Held-out question?",
        "options": str(["right", "wrong 1", "wrong 2", "wrong 3"]), "answer": "A"}])
    previous, _ = build_mix(train, test, [Trace("original_explanations", "expert", "original-id",
        train.iloc[0].question, "An existing expert explanation.", "right")])
    return train, test, previous


def local_row(train, **updates):
    row = {**train.iloc[0].to_dict(), "distill_reasoning": "A new complete reasoning chain.",
        "distill_answer": "a", "distill_ans_correct": True}
    row.update(updates)
    return pd.DataFrame([row])


def traces(frame, source="student_llama_3b"):
    return collect_local_traces(frame, source, source)


def test_preserves_every_inherited_row_and_keeps_different_student_solutions(dataset):
    train, test, previous = dataset
    additions = traces(local_row(train)) + traces(local_row(train,
        distill_reasoning="A second student's different solution."), "student_phi4_mini")
    out, audit = extend_mix(train, test, previous, additions)
    assert len(out) == 3 and audit.status.eq("included").all()
    pd.testing.assert_frame_equal(out.set_index("question_id").loc[previous.question_id],
        previous.set_index("question_id"))
    assert out.question_id.is_unique and out.original_question_id.eq("17").all()


def test_excludes_test_question_even_with_a_training_id(dataset):
    train, test, previous = dataset
    malicious = local_row(train, question=test.iloc[0].question, question_id="17")
    out, audit = extend_mix(train, test, previous, traces(malicious))
    assert len(out) == 1 and audit.status.iloc[-1] == "test_question"


@pytest.mark.parametrize("changes,status", [
    ({"distill_answer": "b"}, "predicted_option_disagrees_with_our_gold"),
    ({"distill_ans_correct": False}, "source_marked_incorrect"),
    ({"answer": "B", "distill_answer": "b"}, "source_gold_disagrees_with_our_gold"),
    ({"distill_reasoning": "a"}, "answer_only_no_reasoning"),
    ({"distill_reasoning": "b"}, "answer_only_no_reasoning"),
    ({"distill_reasoning": ""}, "empty_reasoning"),
    ({"distill_reasoning": "<think>embedded control</think>"}, "embedded_chat_or_thinking_control_token"),
    ({"corrected_reasoning": "Actually choose A."}, "teacher_correction_excluded"),
])
def test_rejects_invalid_additions_without_trusting_correctness_flags(dataset, changes, status):
    train, test, previous = dataset
    out, audit = extend_mix(train, test, previous, traces(local_row(train, **changes)))
    assert len(out) == 1 and audit.status.iloc[-1] == status


def test_keeps_shuffled_option_order_and_checks_the_selected_option_text(dataset):
    train, test, previous = dataset
    frame = local_row(train, options=str(["wrong 1", "right", "wrong 2", "wrong 3"]),
        answer="B", distill_answer="b", distill_reasoning="Option B is correct after comparing the alternatives.")
    out, audit = extend_mix(train, test, previous, traces(frame, "mistral_large_4_0"))
    added = out[out.source.eq("mistral_large_4_0")].iloc[0]
    assert added.option_order_changed and added.answer == "B" and added.original_answer == "A"
    assert option_list(added.options)[1] == "right"
    assert audit.status.eq("included").all()


def test_budget_applies_to_full_target_without_truncation(dataset):
    train, test, previous = dataset
    additions = traces(local_row(train, distill_reasoning="x" * 84))
    additions += traces(local_row(train, distill_reasoning="y" * 85), "student_qwen_3b")
    out, audit = extend_mix(train, test, previous, additions, max_chars=100)
    assert len(out) == 2
    assert out.loc[out.source.eq("student_llama_3b"), "distill_reasoning"].tolist() == ["x" * 84]
    assert out.response_chars.max() == 100
    assert audit.status.tolist() == ["included", "included", "over_character_budget"]


def test_deduplicates_exact_targets_but_preserves_option_order_and_different_chains(dataset):
    train, test, previous = dataset
    same = traces(local_row(train, distill_reasoning=previous.iloc[0].distill_reasoning))
    distinct = traces(local_row(train))
    repeated = traces(local_row(train), "student_phi4_mini")
    reordered = traces(local_row(train, options=str(["wrong 1", "right", "wrong 2", "wrong 3"]),
        answer="B", distill_answer="b"), "student_qwen_3b")
    out, audit = extend_mix(train, test, previous, same + distinct + repeated + reordered)
    assert len(out) == 3
    assert audit.status.tolist() == ["included", "duplicate_training_target", "included",
        "duplicate_training_target", "included"]


def test_refuses_to_silently_drop_invalid_inherited_rows(dataset):
    train, test, previous = dataset
    previous.loc[0, "question"] = test.iloc[0].question
    with pytest.raises(ValueError, match="Previous mixture fails current checks"):
        extend_mix(train, test, previous, traces(local_row(train)))


def test_refuses_to_reintroduce_previous_teacher_corrections(dataset):
    train, test, previous = dataset
    previous.loc[0, "source"] = "current_corrected"
    previous.loc[0, "source_variant"] = "corrected_concat"
    with pytest.raises(ValueError, match="exclude appended teacher corrections"):
        extend_mix(train, test, previous, [])
