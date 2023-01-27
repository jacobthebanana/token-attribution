import os
import unittest

import datasets
from transformers import AutoTokenizer

from ..data.dataloaders import get_dataloader

EXAMPLE_HF_DATASET = os.environ.get("EXAMPLE_HF_DATASET", "data/processed/reddit_eli5")
EXAMPLE_HF_TOKENIZER = os.environ.get(
    "EXAMPLE_HF_TOKENIZER", "Hello-SimpleAI/chatgpt-detector-roberta"
)


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.load_from_disk(EXAMPLE_HF_DATASET)
        self.tokenizer = AutoTokenizer.from_pretrained(EXAMPLE_HF_TOKENIZER)

    def test_dataloader_output(self):
        init_dataloader, num_steps = get_dataloader(
            EXAMPLE_HF_DATASET, "train", self.tokenizer, 1.0, 16, 128, 1
        )
        dataloader = init_dataloader()
        for batch in dataloader:
            break
