# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DataCollatorForLanguageModeling compat patch.

The patch lives in ``opake.transformers.patches._data_patches`` and is applied
automatically on ``import opake.transformers``. This test module imports the
package at collection time (via the parent conftest) so the patch is live.
"""

import pytest


class TestDataCollatorPatch:
    def test_empty_after_nonempty_returns_learned_structure(self):
        transformers = pytest.importorskip("transformers")
        DataCollatorForLanguageModeling = transformers.DataCollatorForLanguageModeling
        AutoTokenizer = transformers.AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # First call: non-empty, learns template
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        nonempty_result = collator(examples)

        # Second call: empty, returns learned structure
        result = collator([])
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape[0] == 0
        assert result["labels"].shape[0] == 0
        assert result["input_ids"].dtype == nonempty_result["input_ids"].dtype
        assert result["input_ids"].shape[1:] == nonempty_result["input_ids"].shape[1:]

    def test_nonempty_examples_unchanged(self):
        transformers = pytest.importorskip("transformers")
        DataCollatorForLanguageModeling = transformers.DataCollatorForLanguageModeling
        AutoTokenizer = transformers.AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        result = collator(examples)

        assert result["input_ids"].shape[0] == 2
        assert result["labels"].shape[0] == 2
