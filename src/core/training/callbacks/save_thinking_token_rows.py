"""Callback that dumps the <think>/</think> rows of embed_tokens (and lm_head if
untied) into each checkpoint dir alongside the adapter. Needed because PEFT's
save_pretrained writes only adapter weights, while these rows live in the base
model and would otherwise be lost.

Loaded back at eval time by MultiCheckpointEvaluator after attaching the adapter.
"""

from pathlib import Path
from typing import override

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import PreTrainedModel

ROWS_FILENAME = "thinking_token_rows.pt"


def write_thinking_token_rows(model: PreTrainedModel, new_ids: list[int], target_dir: Path) -> Path:
    """Persist the `new_ids` rows of embed_tokens (and lm_head if untied) into target_dir.

    PEFT's save_pretrained writes only adapter weights, while these trained rows live in
    the base model and would otherwise be lost. The payload schema ({new_ids, input_rows,
    [output_rows]}) is what the eval-time loader reapplies after attaching the adapter.
    Single-sourced here so the checkpoint callback and the resampling-trainer model
    snapshot can't drift.
    """
    ids_t = torch.tensor(new_ids, dtype=torch.long)
    in_w: torch.Tensor = model.get_input_embeddings().weight.detach()  # type: ignore[assignment]
    in_rows = in_w[ids_t].cpu().clone()

    payload: dict = {
        "new_ids": list(new_ids),
        "input_rows": in_rows,
    }

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if not tied:
        out_layer = model.get_output_embeddings()
        if out_layer is not None:
            out_w: torch.Tensor = out_layer.weight.detach()  # type: ignore[assignment]
            payload["output_rows"] = out_w[ids_t].cpu().clone()

    out_path = target_dir / ROWS_FILENAME
    torch.save(payload, out_path)
    return out_path


def restore_thinking_token_rows(model: PreTrainedModel, new_ids: list[int], source_dir: Path) -> None:
    """Write the rows persisted by `write_thinking_token_rows` back into the model in place.

    Needed on resume: Trainer restores the adapter and optimizer state from the checkpoint,
    but the trained rows live in the base model and would otherwise silently revert to
    their pre-training values.
    """
    rows_path = source_dir / ROWS_FILENAME
    assert rows_path.exists(), f"Resuming from {source_dir} but {ROWS_FILENAME} is missing"
    payload = torch.load(rows_path, map_location="cpu", weights_only=True)
    assert payload["new_ids"] == list(new_ids), (
        f"{rows_path} has new_ids={payload['new_ids']}, expected {list(new_ids)}"
    )

    with torch.no_grad():
        in_w: torch.Tensor = model.get_input_embeddings().weight  # type: ignore[assignment]
        ids_t = torch.tensor(new_ids, dtype=torch.long, device=in_w.device)
        in_w.data[ids_t] = payload["input_rows"].to(dtype=in_w.dtype, device=in_w.device)
        if "output_rows" in payload:
            out_layer = model.get_output_embeddings()
            assert out_layer is not None, f"{rows_path} has output_rows but model has no output embedding layer"
            out_w: torch.Tensor = out_layer.weight  # type: ignore[assignment]
            out_w.data[ids_t] = payload["output_rows"].to(dtype=out_w.dtype, device=out_w.device)


class SaveThinkingTokenRowsCallback(TrainerCallback):
    def __init__(self, new_ids: list[int]):
        assert new_ids, "new_ids must be non-empty"
        self.new_ids = new_ids

    @override
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        model: PreTrainedModel = kwargs["model"]
        # Trainer just wrote checkpoint-<global_step>. Match its naming exactly.
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        write_thinking_token_rows(model, self.new_ids, checkpoint_dir)
