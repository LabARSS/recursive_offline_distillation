"""Leakage, scientific label alignment, and length checks for public GPQA traces."""

import pandas as pd
import pytest

from src.preprocessing.build_gpqa_public_reasoning_mix import (
    Trace,
    build_mix,
    collect_traces,
    normalize,
    option_list,
    parse_original_answer,
    parse_sxiong_question,
)
from src.preprocessing.download_gpqa_public_traces import SXIONG_COLUMNS


@pytest.fixture
def split_frames():
    train = pd.DataFrame([{"question_id": "17", "question": "A scientific training problem?",
        "options": str(["right", "wrong 1", "wrong 2", "wrong 3"]), "answer": "A"}])
    test = pd.DataFrame([{"question_id": "42", "question": "A held-out scientific problem?",
        "options": str(["right", "wrong 1", "wrong 2", "wrong 3"]), "answer": "A"}])
    return train, test


def candidate(question="A scientific training problem?", **kwargs):
    values = dict(source="original_explanations", variant="expert", record_id="foreign-id",
        question=question, reasoning="A complete, independently stated explanation.", gold_text="right")
    values.update(kwargs)
    return Trace(**values)


def test_excludes_test_question_despite_train_split_and_train_looking_id(split_frames):
    train, test = split_frames
    evil = candidate(test.iloc[0].question, record_id="17", split="train")
    out, audit = build_mix(train, test, [candidate(), evil])
    assert len(out) == 1
    assert out.original_question_id.tolist() == ["17"]
    assert audit.status.tolist() == ["included", "test_question"]


def test_preserves_external_option_order_and_its_answer_letters(split_frames):
    train, test = split_frames
    trace = candidate(source="sxiong_written", options=["wrong 2", "right", "wrong 3", "wrong 1"],
        source_answer="B", written_answer="B", reasoning="Option B follows from the calculation.")
    out, audit = build_mix(train, test, [trace])
    row = out.iloc[0]
    assert option_list(row.options)[1] == "right"
    assert row.answer == "B" and row.distill_answer == "b"
    assert row.original_answer == "A" and row.option_order_changed
    assert row.distill_reasoning == trace.reasoning
    assert audit.status.tolist() == ["included"]


@pytest.mark.parametrize("kind,expected", [
    ("gold", "source_gold_disagrees_with_our_gold"),
    ("options", "source_option_set_differs"),
    ("written", "written_answer_disagrees_or_missing"),
    ("incorrect", "source_marked_incorrect"),
    ("selected", "predicted_option_disagrees_with_our_gold"),
])
def test_rejects_inconsistent_or_incorrect_sources(split_frames, kind, expected):
    train, test = split_frames
    trace = candidate(options=option_list(train.iloc[0].options), source_answer="A", written_answer="A")
    if kind == "gold": trace.gold_text = "wrong 1"
    if kind == "options": trace.options[-1] = "a replacement distractor"
    if kind == "written": trace.written_answer = "D"
    if kind == "incorrect": trace.source_correct = False
    if kind == "selected": trace.source_answer = trace.written_answer = "D"
    out, audit = build_mix(train, test, [candidate(), trace])
    assert len(out) == 1
    assert audit.status.iloc[1] == expected


def test_scientific_symbol_case_is_preserved(split_frames):
    assert normalize("q/L^2") != normalize("q/l^2")
    train, test = split_frames
    trace = candidate(gold_text="RIGHT")
    out, audit = build_mix(train, test, [candidate(), trace])
    assert len(out) == 1 and audit.status.iloc[1] == "source_gold_disagrees_with_our_gold"


def test_duplicate_option_text_does_not_change_the_source_letter(split_frames):
    train, test = split_frames
    train.loc[0, "options"] = str(["right", "right", "wrong 2", "wrong 3"])
    train.loc[0, "answer"] = "B"
    out, _ = build_mix(train, test, [candidate()])
    assert out.iloc[0].distill_answer == "b"


def test_budget_counts_complete_target_and_never_truncates(split_frames):
    train, test = split_frames
    short = candidate(reasoning="x" * 20)
    long = candidate(reasoning="y" * 21)
    out, audit = build_mix(train, test, [short, long], max_chars=36)
    assert out.distill_reasoning.tolist() == ["x" * 20]
    assert out.response_chars.tolist() == [36]
    assert audit.status.tolist() == ["included", "over_character_budget"]


@pytest.mark.parametrize("reasoning", ["", " ", "<think>nested</think>", "reason<|end|>"])
def test_rejects_empty_or_embedded_control_tokens(split_frames, reasoning):
    train, test = split_frames
    out, audit = build_mix(train, test, [candidate(), candidate(reasoning=reasoning)])
    assert len(out) == 1 and audit.status.iloc[1] != "included"


def test_deduplicates_targets_and_keeps_unique_trace_ids(split_frames):
    train, test = split_frames
    first = candidate()
    second = candidate(reasoning="A distinct valid explanation.")
    out, audit = build_mix(train, test, [first, first, second])
    assert len(out) == 2 and out.question_id.is_unique
    assert out.original_question_id.tolist() == ["17", "17"]
    assert "duplicate_training_target" in audit.status.tolist()


def test_refuses_ambiguous_local_split(split_frames):
    train, test = split_frames
    test.loc[0, "question"] = train.iloc[0].question
    with pytest.raises(ValueError, match="train/test question overlap"):
        build_mix(train, test, [candidate()])


def test_external_parser_handles_choice_letters_inside_question():
    question = "Q: Analyze compounds:\nA. compound one\nB. compound two\nChoices:\n  A. first\n  B. second\n  C. third\n  D. fourth"
    stem, options = parse_sxiong_question(question)
    assert stem == "Analyze compounds:\nA. compound one\nB. compound two"
    assert options == ["first", "second", "third", "fourth"]
    with pytest.raises(ValueError):
        parse_sxiong_question("Q: Problem\nChoices:\nA. one\nB. two")


def test_original_explanation_parsing():
    assert parse_original_answer("<think>A complete explanation.</think>\n\n#### correct text") == (
        "A complete explanation.", "correct text")


def test_sxiong_download_projection_and_loader_ignore_hidden_reasoning():
    assert "reasoning_content" not in SXIONG_COLUMNS
    raw = {"id": "source-1", "question": "Q: Problem?\nChoices:\nA. right\nB. wrong 1\nC. wrong 2\nD. wrong 3",
        "answer": "A", "content": "Written derivation. \\boxed{A}", "correct": True,
        "source_split": "train", "model": "deepseek-r1", "reasoning_content": "HIDDEN SHOULD NEVER BE USED"}
    traces = collect_traces(pd.DataFrame(), pd.DataFrame(), pd.DataFrame([raw]), pd.DataFrame())
    assert len(traces) == 1 and traces[0].reasoning == raw["content"]
    assert traces[0].written_answer == "A"
    assert "HIDDEN" not in traces[0].reasoning
