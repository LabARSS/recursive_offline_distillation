"""Answer extraction and artifact contract for the MATH downloader."""

import pandas as pd
import pytest

from src.preprocessing import download_hendrycks_math as downloader


@pytest.mark.parametrize(
    ("solution", "expected"),
    [
        (r"The answer is $\boxed{42}$.", "42"),
        (r"The answer is $\boxed{  -3  }$.", "-3"),
        (r"$\boxed{\frac{1}{\sqrt{2}}}$", r"\frac{1}{\sqrt{2}}"),
        (r"$\boxed{\{1, 2\}}$", r"\{1, 2\}"),
        (r"$\boxed{2 \text{ euros}}$", r"2 \text{ euros}"),
        (r"First $\boxed{1}$, finally $\boxed{2}$.", "2"),
        (r"The answer is $\boxed { (3, -1) }$.", "(3, -1)"),
        (r"The answer is $\boxed 12$.", "12"),
        (r"$\boxed \frac{1}{2}$", r"\frac{1}{2}"),
        (r"First $\fbox{1}$, finally $\fbox{\frac{2}{3}}$.", r"\frac{2}{3}"),
        (r"$\boxed{1}$, followed by $\fbox{2}$.", "1"),
    ],
)
def test_extract_final_answer(solution, expected):
    assert downloader.extract_final_answer(solution) == expected


@pytest.mark.parametrize(
    "solution",
    [
        "No final answer here.",
        r"\boxed",
        "\\boxed \n",
        r"\boxed{}",
        r"\boxed{ }",
        r"\boxed{\frac{1}{2}",
        r"\boxed 12",
        r"\boxed $",
        r"$\boxed{1}$, finally \boxed{2",
    ],
)
def test_extract_final_answer_rejects_missing_or_malformed_answers(solution):
    with pytest.raises(ValueError):
        downloader.extract_final_answer(solution)


def test_main_writes_both_splits_in_gsm8k_format(tmp_path, monkeypatch):
    calls = []

    def fake_load_dataset(dataset_id, config):
        calls.append((dataset_id, config))
        return {
            split: [
                {
                    "problem": f"{split}/{config}/{idx}",
                    "solution": f"Worked solution for {config}: $\\boxed{{{idx}}}$.",
                    "level": "Level 1",
                    "type": config,
                }
                for idx in range(2)
            ]
            for split in ("train", "test")
        }

    output_dir = tmp_path / "new" / "source" / "hendrycks_math"
    monkeypatch.setattr(downloader, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(downloader, "OUTPUT_DIR", output_dir)
    monkeypatch.chdir(tmp_path)

    downloader.main()

    assert calls == [
        (downloader.DATASET_ID, config)
        for config in downloader.DATASET_CONFIGS
    ]
    for split in ("train", "test"):
        df = pd.read_parquet(output_dir / f"hendrycks_math_{split}.parquet")
        assert list(df.columns) == ["question", "answer", "answer_w_steps", "question_id"]
        assert len(df) == 2 * len(downloader.DATASET_CONFIGS)
        assert df["question_id"].tolist() == [str(idx) for idx in range(len(df))]
        assert df["question_id"].is_unique
        assert df["question"].tolist() == [
            f"{split}/{config}/{idx}"
            for config in downloader.DATASET_CONFIGS
            for idx in range(2)
        ]
        assert df["answer"].tolist() == ["0", "1"] * len(downloader.DATASET_CONFIGS)
        assert df["answer_w_steps"].tolist() == [
            f"Worked solution for {config}: $\\boxed{{{idx}}}$."
            for config in downloader.DATASET_CONFIGS
            for idx in range(2)
        ]
        for column in df.columns:
            assert df[column].map(lambda value: isinstance(value, str)).all()


def test_main_reports_split_subject_and_row_for_invalid_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "DATASET_CONFIGS", ("algebra",))
    monkeypatch.setattr(downloader, "SPLITS", ("test",))
    monkeypatch.setattr(downloader, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(
        downloader,
        "load_dataset",
        lambda *args, **kwargs: {
            "test": [
                {"problem": "Valid", "solution": r"$\boxed{1}$"},
                {"problem": "Invalid", "solution": "No boxed answer"},
            ]
        },
    )

    with pytest.raises(ValueError, match="test/algebra row 1"):
        downloader.main()

    assert not (tmp_path / "output" / "hendrycks_math_test.parquet").exists()


def test_main_corrects_known_empty_answers_and_preserves_solutions(tmp_path, monkeypatch, capsys):
    solutions = list(downloader.SOURCE_ANSWER_CORRECTIONS)
    monkeypatch.setattr(downloader, "DATASET_CONFIGS", ("number_theory",))
    monkeypatch.setattr(downloader, "SPLITS", ("train",))
    monkeypatch.setattr(downloader, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        downloader,
        "load_dataset",
        lambda *args, **kwargs: {
            "train": [{"problem": "Count primes", "solution": solution} for solution in solutions]
        },
    )

    downloader.main()

    df = pd.read_parquet(tmp_path / "hendrycks_math_train.parquet")
    assert df["answer"].tolist() == ["0", "0"]
    assert df["answer_w_steps"].tolist() == solutions
    assert capsys.readouterr().out.count("Corrected empty answer") == 2


@pytest.mark.parametrize(
    ("split", "config", "suffix"),
    [("test", "number_theory", ""), ("train", "algebra", ""), ("train", "number_theory", " ")],
)
def test_main_does_not_correct_other_empty_answers(tmp_path, monkeypatch, split, config, suffix):
    solution = next(iter(downloader.SOURCE_ANSWER_CORRECTIONS)) + suffix
    monkeypatch.setattr(downloader, "DATASET_CONFIGS", (config,))
    monkeypatch.setattr(downloader, "SPLITS", (split,))
    monkeypatch.setattr(downloader, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        downloader,
        "load_dataset",
        lambda *args, **kwargs: {split: [{"problem": "Count primes", "solution": solution}]},
    )

    with pytest.raises(ValueError, match="Boxed answer is empty"):
        downloader.main()

    assert not (tmp_path / f"hendrycks_math_{split}.parquet").exists()
