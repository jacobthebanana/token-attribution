import argparse
import datetime
import os
import socket
from typing import Callable, Dict, Iterator, Literal, Optional, Tuple, Union, Any

import chex
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    FlaxAutoModelForTokenClassification,
    FlaxRobertaForSequenceClassification,
)
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
from sklearn.metrics import confusion_matrix

import wandb

from ..data.dataloaders import Batch, get_dataloader

Params = Dict
Gradients = Dict
Loss = jnp.ndarray
CorrectRatio = jnp.ndarray
Predictions = jnp.ndarray
EvalStats = Dict[Literal["loss", "accuracy", "confusion_matrix"], Any]


def loss_accuracy_fn(
    batch: Batch, params: Dict, apply_fn: Callable
) -> Tuple[Loss, Tuple[CorrectRatio, Predictions]]:
    """
    Return mean squared error between labels and
    the sum of output logits over sequence,
    as well as prediction accuracy if split at 0.5.

    """
    model_output = apply_fn(batch.input_ids, batch.attention_mask, params=params)

    assert isinstance(model_output, FlaxSequenceClassifierOutput)
    output_logits = model_output.logits  # (batch, class)
    per_sequence_loss = optax.softmax_cross_entropy_with_integer_labels(
        output_logits, batch.labels
    )

    predictions = jnp.argmax(output_logits, axis=-1).reshape((-1,))
    assert predictions.shape == batch.labels.shape, (
        predictions.shape,
        batch.labels.shape,
    )
    num_correct = jnp.sum(predictions == batch.labels)
    correct_ratio = num_correct / len(batch.labels)

    loss = jnp.mean(per_sequence_loss)
    return loss, (correct_ratio, predictions)


def loss_grad_fn(
    batch: Batch, params: Union[Dict, FrozenDict], apply_fn: Callable
) -> Tuple[Tuple[Loss, Tuple[CorrectRatio, Predictions]], Gradients]:
    ...


loss_grad_fn = jax.value_and_grad(loss_accuracy_fn, has_aux=True, argnums=1)


def train_step(
    batch: Batch, state: TrainState
) -> Tuple[TrainState, Tuple[Loss, CorrectRatio]]:
    (loss, (accuracy, _)), gradients = loss_grad_fn(batch, state.params, state.apply_fn)
    new_state = state.apply_gradients(grads=gradients)
    return new_state, (loss, accuracy)


def jit_train_step(
    batch: Batch, state: TrainState
) -> Tuple[TrainState, Tuple[Loss, CorrectRatio]]:
    ...


jit_train_step = jax.jit(train_step, donate_argnums=1)  # type: ignore


def jit_loss_accuracy_fn(
    batch: Batch, params: FrozenDict, apply_fn: Callable
) -> Tuple[Loss, Tuple[CorrectRatio, Predictions]]:
    ...


jit_loss_accuracy_fn = jax.jit(loss_accuracy_fn, static_argnames=["apply_fn"])


def evaluate_model(
    apply_fn: Callable,
    params: FrozenDict,
    dataloader: Iterator[Batch],
    num_eval_steps: Optional[int] = None,
) -> EvalStats:
    loss_tally = 0.0
    accuracy_tally = 0.0
    num_batches = 0

    all_labels = []
    all_predictions = []

    for batch in tqdm(
        dataloader, ncols=80, desc="Evaluating", leave=False, total=num_eval_steps
    ):
        loss, (accuracy, predictions) = jit_loss_accuracy_fn(batch, params, apply_fn)
        loss_tally += loss.item()
        accuracy_tally += accuracy.item()
        num_batches += 1

        all_labels.extend(batch.labels.flatten().tolist())
        all_predictions.extend(predictions.flatten().tolist())

    stats: EvalStats = {
        "loss": loss_tally / num_batches,
        "accuracy": accuracy_tally / num_batches,
        "confusion_matrix": confusion_matrix(all_labels, all_predictions),
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--learning_rate_min", type=float, default=-1)
    parser.add_argument("--regularization", type=float, default=0)
    parser.add_argument("--base_hf_model")
    parser.add_argument("--hf_dataset_path")
    parser.add_argument("--hf_tokenizer", required=False, default=None)
    parser.add_argument("--num_epochs", type=float, default=1)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--eval_subsample_multiplier", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--early_stop_threshold",
        type=int,
        default=6,
        help="Number of steps compared when early stopping.",
    )

    args = parser.parse_args()
    learning_rate: float = args.learning_rate
    learning_rate_min: float = args.learning_rate_min
    regularization: float = args.regularization
    base_hf_model: str = args.base_hf_model
    hf_dataset_path: str = args.hf_dataset_path
    hf_tokenizer: Optional[str] = args.hf_tokenizer
    train_batch_size: int = args.train_batch_size
    max_length: int = args.max_length
    num_epochs: float = args.num_epochs
    eval_every: int = args.eval_every
    eval_subsample_multiplier: int = args.eval_subsample_multiplier
    early_stop_threshold: int = args.early_stop_threshold

    wandb_run_name = datetime.datetime.now().isoformat() + "-" + socket.gethostname()
    wandb.init(name=wandb_run_name)
    wandb.config.update(args.__dict__)

    if learning_rate_min < 0:
        learning_rate_min = learning_rate

    if hf_tokenizer is None:
        hf_tokenizer = base_hf_model

    # Data
    tokenizer = AutoTokenizer.from_pretrained(base_hf_model)
    init_train_dataloader, num_train_steps = get_dataloader(
        hf_dataset_path, "train", tokenizer, num_epochs, train_batch_size, max_length
    )
    init_eval_dataloader, num_eval_steps = get_dataloader(
        hf_dataset_path,
        "validation",
        tokenizer,
        1 / eval_subsample_multiplier,
        train_batch_size,
        max_length,
        subsampling_multiplier=eval_subsample_multiplier,
    )

    # Model, Optimization, and (single-host only) sharding.
    model: FlaxRobertaForSequenceClassification
    model = FlaxRobertaForSequenceClassification.from_pretrained(
        base_hf_model, from_pt=True
    )  # type: ignore
    sharding_scheme = jax.sharding.PositionalSharding(jax.devices()).replicate()
    params: FrozenDict = jax.device_put(model.params, sharding_scheme)

    lr_schedule = optax.linear_schedule(
        learning_rate, learning_rate_min, num_train_steps
    )
    optimizer = optax.adamw(lr_schedule)
    opt_state = optimizer.init(model.params)

    opt_state = jax.device_put(opt_state, sharding_scheme)

    train_state = TrainState(0, model.__call__, params, optimizer, opt_state)

    # Training loop
    train_dataloader = init_train_dataloader()
    eval_loss_history = []
    for batch in tqdm(train_dataloader, ncols=80, total=num_train_steps):
        train_state: TrainState
        train_state, (loss, accuracy) = jit_train_step(
            batch, train_state, regularization
        )

        stats = {"train_loss": loss, "train_accuracy": accuracy}

        if train_state.step % eval_every == 1:
            eval_stats = evaluate_model(
                model.__call__,
                train_state.params,
                init_eval_dataloader(),
                num_eval_steps,
            )
            stats = {**stats, **{"validation_" + k: v for k, v in eval_stats.items()}}

            if early_stop_threshold > 0 and len(eval_loss_history) > 0:
                recent_eval_losses = eval_loss_history[-early_stop_threshold:]
                if eval_stats["loss"] > max(recent_eval_losses):
                    wandb.log(stats)
                    wandb.finish()
                    break

            output_path = os.path.join("data/artifacts", wandb.run.id)  # type: ignore
            model.save_pretrained(output_path, params=train_state.params)

        wandb.log(stats)
    wandb.finish()
