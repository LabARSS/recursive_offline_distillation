"""
Build a mixed reasoning-trace training set for distilling Hendrycks MATH into Llama-3.2-3B.

Sources:
  omi2_math     OpenMathInstruct-2 "math" rows (Llama-3.1-405B-Instruct), from
                hendrycks_math_openmathinstruct2_math_all2sol_aug7000.parquet
  qwen3_math8k  aochongoliverli/math8k-sft-Qwen3-32B-16k-reasoning-traces (Qwen3-32B thinking)
  nemotron_v4   nvidia/Nemotron-SFT-Math-v4 CoT rows (DeepSeek-V4-Pro thinking), traces
                under NEMOTRON_MAX_CHARS, NEMOTRON_ROWS sampled
  llama_self    Llama-3.2-3B-Instruct's own correct CoT on MATH train,
                hendrycks_math_self_distilled_llama_3b.parquet

Every source goes through the same Hendrycks MATH test decontamination as
download_openmathinstruct2.py (exact match or a shared 13-word n-gram). MATH-train
sources keep only rows whose answer matches the MATH ground truth. The reasoning is
truncated to TRUNCATE_CHARS characters.

Usage:
  PYTHONPATH=src python src/preprocessing/build_hendrycks_math_reasoning_mix.py

Outputs:
  data/out/distillation/hendrycks_math_reasoning_mix_llama_3b.parquet
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from download_hendrycks_math import extract_final_answer
from download_openmathinstruct2 import NGRAM_SIZE, TEST_PATH, ngrams, normalize
from huggingface_hub import hf_hub_download

from core.utils.cot_answers import verify_math_answer

SEED = 42
TRUNCATE_CHARS = 8192
NEMOTRON_ROWS = 6000
NEMOTRON_MAX_CHARS = 10000
NEMOTRON_SHARDS = [f"data/train-{i:05d}-of-00012.parquet" for i in range(12)]

ROOT = Path(__file__).resolve().parents[2]
DISTILL_DIR = ROOT / "data/out/distillation"
TRAIN_PATH = ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet"
OMI2_PATH = DISTILL_DIR / "hendrycks_math_openmathinstruct2_math_all2sol_aug7000.parquet"
SELF_PATH = DISTILL_DIR / "hendrycks_math_self_distilled_llama_3b.parquet"
OUTPUT_PATH = DISTILL_DIR / "hendrycks_math_reasoning_mix_llama_3b.parquet"
COLUMNS = ["question", "answer", "distill_reasoning", "distill_answer", "source"]


def safe_verify(gold: str, prediction: str) -> bool:
    try:
        return verify_math_answer(str(gold), str(prediction))
    except Exception:
        return False


def load_omi2_math(train_answers: dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(OMI2_PATH)
    df = df[df["problem_source"] == "math"].copy()
    df["answer"] = df["question"].map(lambda q: train_answers[normalize(q)])
    df = df[[safe_verify(g, a) for g, a in zip(df["answer"], df["distill_answer"])]]
    return df.assign(source="omi2_math")[COLUMNS]


def load_qwen3_math8k(train_answers: dict[str, str]) -> pd.DataFrame:
    path = hf_hub_download(
        "aochongoliverli/math8k-sft-Qwen3-32B-16k-reasoning-traces",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = []
    for problem, conversation in pd.read_parquet(path)[["problem", "conversations"]].itertuples(index=False):
        answer = train_answers.get(normalize(problem))
        reply = [m for m in conversation if m["role"] == "assistant"][-1]["content"]
        if answer is None or "</think>" not in reply:
            continue
        reasoning, solution = reply.split("</think>", 1)
        try:
            prediction = extract_final_answer(solution)
        except ValueError:
            continue
        if safe_verify(answer, prediction):
            rows.append(
                {
                    "question": problem,
                    "answer": answer,
                    "distill_reasoning": reasoning.replace("<think>", "").strip(),
                    "distill_answer": prediction,
                }
            )
    return pd.DataFrame(rows).assign(source="qwen3_math8k")[COLUMNS]


def load_nemotron_v4() -> pd.DataFrame:
    rows = []
    for shard in NEMOTRON_SHARDS:
        path = hf_hub_download("nvidia/Nemotron-SFT-Math-v4", shard, repo_type="dataset")
        table = pq.read_table(path, columns=["problem", "expected_answer", "messages", "subset"])
        for problem, answer, messages, subset in zip(*(table.column(c).to_pylist() for c in table.column_names)):
            if subset != "cot":
                continue
            messages = json.loads(messages) if isinstance(messages, str) else messages
            reply = next(m for m in messages if m["role"] == "assistant")
            reasoning = (reply.get("reasoning_content") or "").strip()
            if reasoning and str(answer).strip() and len(reasoning) < NEMOTRON_MAX_CHARS:
                rows.append(
                    {
                        "question": problem,
                        "answer": str(answer).strip(),
                        "distill_reasoning": reasoning,
                        "distill_answer": str(answer).strip(),
                    }
                )
    df = pd.DataFrame(rows).drop_duplicates("question")
    print(f"nemotron_v4: {len(df)} unique CoT problems with reasoning under {NEMOTRON_MAX_CHARS} chars")
    return df.assign(source="nemotron_v4")[COLUMNS]


def load_llama_self() -> pd.DataFrame:
    df = pd.read_parquet(SELF_PATH)
    return df.assign(source="llama_self")[COLUMNS]


def main():
    train = pd.read_parquet(TRAIN_PATH)
    train_answers = dict(zip(train["question"].map(normalize), train["answer"]))
    test_questions = {normalize(q) for q in pd.read_parquet(TEST_PATH)["question"]}
    test_ngrams = set().union(*(ngrams(q) for q in test_questions))
    short_test_questions = {q for q in test_questions if len(q.split(" ")) < NGRAM_SIZE}

    def contaminated(question: str) -> bool:
        q = normalize(question)
        return q in test_questions or q in short_test_questions or not ngrams(q).isdisjoint(test_ngrams)

    subsets = []
    loaders = (
        lambda: load_omi2_math(train_answers),
        lambda: load_qwen3_math8k(train_answers),
        load_nemotron_v4,
        load_llama_self,
    )
    for loader in loaders:
        df = loader()
        dirty = df["question"].map(contaminated)
        source = df["source"].iloc[0]
        df = df[~dirty]
        if source == "nemotron_v4":
            df = df.sample(n=min(NEMOTRON_ROWS, len(df)), random_state=SEED)
        truncated = df["distill_reasoning"].str.len() > TRUNCATE_CHARS
        print(
            f"{source}: dropped {int(dirty.sum())} test matches, kept {len(df)} rows, "
            f"truncating {int(truncated.sum())} reasoning traces to {TRUNCATE_CHARS} chars"
        )
        subsets.append(df)

    out = pd.concat(subsets).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out["distill_reasoning"] = out["distill_reasoning"].str[:TRUNCATE_CHARS]
    out["question_id"] = [str(i) for i in range(len(out))]
    out["distill_ans_correct"] = True
    assert not out["question"].map(normalize).isin(test_questions).any(), "Mix contains a test question"

    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUTPUT_PATH}: {out['source'].value_counts().to_dict()}")
    print(out.groupby("source")["distill_reasoning"].apply(lambda s: s.str.len().median()).to_dict(), "median reasoning chars")


if __name__ == "__main__":
    main()
