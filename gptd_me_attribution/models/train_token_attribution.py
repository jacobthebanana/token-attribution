import argparse
import datetime
import os
import socket
from typing import Callable, Dict, Iterator, Literal, Optional, Tuple, Union

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
    FlaxRobertaForTokenClassification,
)
from transformers.modeling_flax_outputs import FlaxTokenClassifierOutput

import wandb

from ..data.dataloaders import Batch, get_dataloader

Params = Dict
Gradients = Dict
Loss = jnp.ndarray
CorrectRatio = jnp.ndarray
EvalStats = Dict[Literal["loss", "accuracy"], float]


def loss_accuracy_fn(
    batch: Batch, params: Dict, apply_fn: Callable, normalize_by_length: bool
) -> Tuple[Loss, CorrectRatio]:
    """
    Return mean squared error between labels and
    the sum of output logits over sequence,
    as well as prediction accuracy if split at 0.5.

    If normalize_by_length = True, the sequence-level value
    would be normalized by the number of non-padding tokens
    in the sequence.
    """
    model_output = apply_fn(batch.input_ids, batch.attention_mask, params=params)

    assert isinstance(model_output, FlaxTokenClassifierOutput)
    token_values = model_output.logits.squeeze(-1)  # (batch, token)

    # Exclude "CLS" logit from output.
    token_mask = batch.attention_mask.at[:, 0].set(0)
    chex.assert_equal_shape((token_mask, token_values))
    masked_token_values = token_values * token_mask

    per_sequence_value_sum = jnp.sum(masked_token_values, axis=-1)  # (batch,)

    if normalize_by_length:
        per_sequence_num_real_tokens = jnp.sum(batch.attention_mask, axis=1)
        per_sequence_value_reference = (
            per_sequence_num_real_tokens * batch.labels
        )  # (batch,)
    else:
        per_sequence_value_reference = 1000.0 * batch.labels  # (batch,)
    chex.assert_equal_shape((per_sequence_value_sum, per_sequence_value_reference))

    predictions = jnp.where(per_sequence_value_sum > 500, 1, 0)
    num_correct = jnp.sum(predictions == batch.labels)
    correct_ratio = num_correct / len(batch.labels)

    per_sequence_loss = jnp.linalg.norm(
        per_sequence_value_sum - per_sequence_value_reference
    )
    loss = jnp.mean(per_sequence_loss)
    return loss, correct_ratio


def loss_grad_fn(
    batch: Batch,
    params: Union[Dict, FrozenDict],
    apply_fn,
    normalize_by_length: bool,
) -> Tuple[Tuple[Loss, CorrectRatio], Gradients]:
    ...


loss_grad_fn = jax.value_and_grad(loss_accuracy_fn, has_aux=True, argnums=1)


def train_step(
    batch: Batch, state: TrainState, normalize_by_length: bool
) -> Tuple[TrainState, Tuple[Loss, CorrectRatio]]:
    (loss, accuracy), gradients = loss_grad_fn(
        batch, state.params, state.apply_fn, normalize_by_length
    )
    new_state = state.apply_gradients(grads=gradients)
    return new_state, (loss, accuracy)


def jit_train_step(
    batch: Batch, state: TrainState, normalize_by_length: bool
) -> Tuple[TrainState, Tuple[Loss, CorrectRatio]]:
    ...


jit_train_step = jax.jit(
    train_step, static_argnames=["normalize_by_length"], donate_argnums=1
)


def jit_loss_accuracy_fn(
    batch: Batch, params: FrozenDict, apply_fn: Callable, normalize_by_length: bool
) -> Tuple[Loss, CorrectRatio]:
    ...


jit_loss_accuracy_fn = jax.jit(
    loss_accuracy_fn, static_argnames=["apply_fn", "normalize_by_length"]
)


def evaluate_model(
    apply_fn: Callable,
    params: FrozenDict,
    dataloader: Iterator[Batch],
    normalize_by_length: bool = False,
    num_eval_steps: Optional[int] = None,
) -> EvalStats:
    loss_tally = 0.0
    accuracy_tally = 0.0
    num_batches = 0

    for batch in tqdm(
        dataloader, ncols=80, desc="Evaluating", leave=False, total=num_eval_steps
    ):
        loss, accuracy = jit_loss_accuracy_fn(
            batch, params, apply_fn, normalize_by_length
        )
        loss_tally += loss.item()
        accuracy_tally += accuracy.item()
        num_batches += 1

    stats: EvalStats = {
        "loss": loss_tally / num_batches,
        "accuracy": accuracy_tally / num_batches,
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--learning_rate_min", type=float, default=-1)
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
    parser.add_argument(
        "--normalize_by_length",
        type=bool,
        default=False,
        help="Normalize value by number of tokens in example.",
    )

    args = parser.parse_args()
    learning_rate: float = args.learning_rate
    learning_rate_min: float = args.learning_rate_min
    base_hf_model: str = args.base_hf_model
    hf_dataset_path: str = args.hf_dataset_path
    hf_tokenizer: Optional[str] = args.hf_tokenizer
    train_batch_size: int = args.train_batch_size
    max_length: int = args.max_length
    num_epochs: float = args.num_epochs
    eval_every: int = args.eval_every
    eval_subsample_multiplier: int = args.eval_subsample_multiplier
    early_stop_threshold: int = args.early_stop_threshold
    normalize_by_length: bool = args.normalize_by_length

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
    model: FlaxRobertaForTokenClassification
    model = FlaxAutoModelForTokenClassification.from_pretrained(
        base_hf_model, from_pt=True, num_labels=1
    )
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
            batch, train_state, normalize_by_length
        )

        stats = {"train_loss": loss, "train_accuracy": accuracy}

        if train_state.step % eval_every == 1:
            eval_stats = evaluate_model(
                model.__call__,
                train_state.params,
                init_eval_dataloader(),
                normalize_by_length,
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
