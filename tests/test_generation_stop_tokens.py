"""Model EOS handling through the real generation loop, without model downloads."""

from types import SimpleNamespace

import pytest
import torch

from core.evaluation.phased_batch_generator import BatchGenerator


PHI_END = 200020
TOKENIZER_EOS = 199999


@pytest.fixture
def make_generator(monkeypatch):
    # Script the sampled tokens and KV tensors; retain the actual phase loop,
    # slot retirement, carry handling, and truncation accounting.
    monkeypatch.setattr(BatchGenerator, "_install_unpad_cache", lambda self: None)

    def make(scripts, *, generation_eos=None, config_eos=None, max_new_tokens=8):
        model = SimpleNamespace(
            config=SimpleNamespace(
                eos_token_id=config_eos,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=1,
            ),
            generation_config=SimpleNamespace(eos_token_id=generation_eos),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=TOKENIZER_EOS)
        generator = BatchGenerator(
            model, tokenizer, max_new_tokens=max_new_tokens, max_batch_size=2, kv_cache_pin_memory=False
        )

        def prefill(slot, prefill_ids, resume_generated_ids=None):
            script = scripts[slot.prompt_ids[0]]
            slot.generated_ids = [script[0]] if resume_generated_ids is None else list(resume_generated_ids)
            slot.seq_position = len(prefill_ids) + 1
            kv = torch.zeros(1, 1, 1, len(prefill_ids), 1)
            return kv, kv.clone(), len(prefill_ids)

        def decode(batch_indices, slots):
            for index, slot in zip(batch_indices, slots):
                script = scripts[slot.prompt_ids[0]]
                slot.generated_ids.append(script[len(slot.generated_ids)])
                slot.seq_position += 1
                generator._valid_lens[index] += 1

        monkeypatch.setattr(generator, "_prefill", prefill)
        monkeypatch.setattr(generator, "_batched_decode", decode)
        return generator

    return make


@pytest.mark.parametrize("generation_eos", [PHI_END, [PHI_END, TOKENIZER_EOS]])
@pytest.mark.parametrize("prefix", [[], [7]])
def test_stops_at_model_eos_before_tokenizer_eos(make_generator, generation_eos, prefix):
    expected = [*prefix, PHI_END]
    generator = make_generator({1: [*expected, 8, TOKENIZER_EOS]}, generation_eos=generation_eos)
    result = generator.generate([[1]])
    assert result.sequences == [expected]
    assert result.truncated == [False]
    assert result.unfinished == [False]


@pytest.mark.parametrize("generation_eos", [None, PHI_END, [PHI_END, TOKENIZER_EOS]])
def test_tokenizer_eos_remains_a_stop_token(make_generator, generation_eos):
    generator = make_generator({1: [7, TOKENIZER_EOS, 8]}, generation_eos=generation_eos)
    assert generator.generate([[1]]).sequences == [[7, TOKENIZER_EOS]]


def test_falls_back_to_model_config_eos(make_generator):
    generator = make_generator({1: [7, PHI_END, 8]}, config_eos=[PHI_END, TOKENIZER_EOS])
    assert generator.generate([[1]]).sequences == [[7, PHI_END]]


@pytest.mark.parametrize(("max_new_tokens", "carry_cap"), [(8, 2), (2, None)])
def test_eos_at_budget_boundary_finishes_without_truncation_or_carry(make_generator, max_new_tokens, carry_cap):
    generator = make_generator({1: [7, PHI_END, 8]}, generation_eos=[PHI_END], max_new_tokens=max_new_tokens)
    result = generator.generate([[1]], carry_at_new_tokens=carry_cap)
    assert result.sequences == [[7, PHI_END]]
    assert result.num_truncated == 0
    assert result.truncated == [False]
    assert result.unfinished == [False]


def test_stopping_one_slot_keeps_other_slots_running(make_generator):
    generator = make_generator(
        {1: [7, PHI_END, 9], 2: [7, 8, TOKENIZER_EOS, 9]}, generation_eos=[PHI_END, TOKENIZER_EOS]
    )
    result = generator.generate([[1], [2]])
    assert result.sequences == [[7, PHI_END], [7, 8, TOKENIZER_EOS]]
    assert result.truncated == [False, False]
    assert result.unfinished == [False, False]
