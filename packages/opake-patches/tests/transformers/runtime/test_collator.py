import pytest

pytest.importorskip("transformers")
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from opake.patches import apply_runtime_patches


def test_collator_empty_batch():
    apply_runtime_patches(empty_batches=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Must first pass a non-empty batch so empty_collate learns the structure
    non_empty_batch = [{"input_ids": [1, 2, 3]}]
    out1 = collator.torch_call(non_empty_batch)
    assert "input_ids" in out1

    # Empty batch
    empty_batch = []

    # Should not throw IndexError/ValueError
    out2 = collator.torch_call(empty_batch)

    assert isinstance(out2, dict)
    assert "input_ids" in out2
    assert out2["input_ids"].numel() == 0
