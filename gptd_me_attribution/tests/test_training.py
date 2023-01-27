import os
import unittest

import chex
import datasets
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax.sharding import PositionalSharding
from transformers import AutoTokenizer, FlaxRobertaForTokenClassification

from ..data.dataloaders import get_dataloader
from ..models.train_token_attribution import (jit_train_step, loss_accuracy_fn,
                                              loss_grad_fn)


def print_shape(tree):
    print(jax.tree_util.tree_map(jnp.shape, tree))


EXAMPLE_HF_DATASET = os.environ.get("EXAMPLE_HF_DATASET", "data/processed/reddit_eli5")
EXAMPLE_HF_MODEL = os.environ.get(
    "EXAMPLE_HF_MODEL", "Hello-SimpleAI/chatgpt-detector-roberta"
)
EXAMPLE_NUM_STEPS = int(os.environ.get("EXAMPLE_NUM_STEPS", "12"))


class BatchedTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model: FlaxRobertaForTokenClassification
        cls.model = FlaxRobertaForTokenClassification.from_pretrained(
            EXAMPLE_HF_MODEL, from_pt=True, num_labels=1
        )  # type: ignore

    def setUp(self):
        self.dataset = datasets.load_from_disk(EXAMPLE_HF_DATASET)
        self.tokenizer = AutoTokenizer.from_pretrained(EXAMPLE_HF_MODEL)

        self.init_dataloader, num_steps = get_dataloader(
            EXAMPLE_HF_DATASET, "train", self.tokenizer, 1.0, 16, 128, 1
        )
        self.model = BatchedTrainingTest.model

        self.params: FrozenDict = self.model.params  # type: ignore

        dataloader = self.init_dataloader()
        self.example_batch = next(dataloader)

    def test_loss_accuracy_fn(self):
        dataloader = self.init_dataloader()
        example_batch = next(dataloader)

        loss, accuracy = loss_accuracy_fn(
            example_batch,
            self.model.params,  # type: ignore
            self.model.__call__,
            False,
        )

    def test_gradient(self):

        (loss, accuracy), gradient = loss_grad_fn(
            self.example_batch, self.params, self.model.__call__, False
        )

        chex.assert_tree_all_equal_shapes(self.params, gradient)

    def test_train_step(self):
        optimizer = optax.adamw(0.001)
        opt_state = optimizer.init(self.params)
        sharding = PositionalSharding(jax.devices()).replicate()
        train_state = TrainState(
            step=0,
            apply_fn=self.model.__call__,
            params=jax.device_put(self.params, sharding),
            tx=optimizer,
            opt_state=jax.device_put(opt_state, sharding),
        )

        loss_history = []
        accuracy_history = []

        for _ in range(EXAMPLE_NUM_STEPS):
            train_state, (loss, accuracy) = jit_train_step(
                self.example_batch, train_state, False
            )

            loss_history.append(loss.item())
            accuracy_history.append(accuracy.item())

        print(loss_history)
        print(accuracy_history)

        optimizer = optax.adamw(0.0001)
        opt_state = optimizer.init(self.params)
        train_state = TrainState(
            step=0,
            apply_fn=self.model.__call__,
            params=jax.device_put(self.params, sharding),
            tx=optimizer,
            opt_state=jax.device_put(opt_state, sharding),
        )

        loss_history = []
        accuracy_history = []

        for _ in range(EXAMPLE_NUM_STEPS):
            train_state, (loss, accuracy) = jit_train_step(
                self.example_batch, train_state, True
            )

            loss_history.append(loss.item())
            accuracy_history.append(accuracy.item())

        print(loss_history)
        print(accuracy_history)
