from __future__ import annotations

from pathlib import Path

import numpy as np

from baseline_skipgram import (
    DEFAULT_CORPUS_PATH,
    choose_demo_pair,
    format_array,
    plot_losses,
    prepare_dataset,
    sigmoid,
)


def build_negative_sampling_distribution(vocab: list[str], counts: dict[str, int], power: float = 0.75) -> np.ndarray:
    weights = np.array([counts[word] ** power for word in vocab], dtype=np.float64)
    return weights / weights.sum()


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0, init_scale: float = 0.01):
        np.random.seed(seed)
        self.V = vocab_size
        self.d = embed_dim
        self.W_in = np.random.randn(vocab_size, embed_dim) * init_scale
        self.W_out = np.random.randn(embed_dim, vocab_size) * init_scale

    def sample_negative_indices(
        self,
        positive_idx: int,
        num_negative: int,
        distribution: np.ndarray,
        rng: np.random.RandomState,
    ) -> list[int]:
        negatives: list[int] = []
        while len(negatives) < num_negative:
            candidate = int(rng.choice(self.V, p=distribution))
            if candidate == positive_idx:
                continue
            negatives.append(candidate)
        return negatives

    def pair_loss_and_gradients(
        self,
        center_idx: int,
        context_idx: int,
        negative_indices: list[int],
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        v_c = self.W_in[center_idx].copy()
        u_o = self.W_out[:, context_idx].copy()
        pos_score = float(np.dot(u_o, v_c))
        pos_sigmoid = float(sigmoid(pos_score))
        loss = -np.log(pos_sigmoid + 1e-12)

        grad_v = (pos_sigmoid - 1.0) * u_o
        grad_W_out = np.zeros_like(self.W_out)
        grad_W_out[:, context_idx] += (pos_sigmoid - 1.0) * v_c

        for negative_idx in negative_indices:
            u_k = self.W_out[:, negative_idx].copy()
            neg_score = float(np.dot(u_k, v_c))
            neg_sigmoid = float(sigmoid(neg_score))
            loss += -np.log(1.0 - neg_sigmoid + 1e-12)
            grad_v += neg_sigmoid * u_k
            grad_W_out[:, negative_idx] += neg_sigmoid * v_c

        grad_W_in = np.zeros_like(self.W_in)
        grad_W_in[center_idx] = grad_v
        return float(loss), grad_W_in, grad_W_out, v_c

    def update_pair(
        self,
        center_idx: int,
        context_idx: int,
        negative_indices: list[int],
        lr: float,
    ) -> dict[str, object]:
        loss, grad_W_in, grad_W_out, v_c = self.pair_loss_and_gradients(center_idx, context_idx, negative_indices)
        self.W_in -= lr * grad_W_in
        self.W_out -= lr * grad_W_out
        return {
            "loss": loss,
            "v_c": v_c,
            "grad_W_in": grad_W_in[center_idx].copy(),
            "grad_W_out_context": grad_W_out[:, context_idx].copy(),
            "updated_center": self.W_in[center_idx].copy(),
            "updated_context_out": self.W_out[:, context_idx].copy(),
        }


def train_negative_sampling(
    model: SkipGramNegativeSampling,
    pairs: list[tuple[int, int]],
    distribution: np.ndarray,
    epochs: int,
    lr_init: float,
    lr_decay: float,
    num_negative: int,
    shuffle_seed: int = 0,
) -> list[float]:
    rng = np.random.RandomState(shuffle_seed)
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        lr = lr_init / (1.0 + lr_decay * epoch)
        shuffled_indices = rng.permutation(len(pairs))
        total_loss = 0.0
        for pair_idx in shuffled_indices:
            center_idx, context_idx = pairs[pair_idx]
            negatives = model.sample_negative_indices(context_idx, num_negative, distribution, rng)
            update = model.update_pair(center_idx, context_idx, negatives, lr)
            total_loss += float(update["loss"])
        losses.append(total_loss / len(pairs))
    return losses


def run_negative_sampling_experiment(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_dir: Path | None = None,
    min_count: int = 1,
    embed_dim: int = 50,
    window_size: int = 2,
    epochs: int = 10,
    lr_init: float = 0.025,
    lr_decay: float = 0.0,
    num_negative: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    data = prepare_dataset(
        corpus_path=corpus_path,
        max_sentences=None,
        min_count=min_count,
        window_size=window_size,
    )
    distribution = build_negative_sampling_distribution(data["vocab"], data["counts"])
    demo_center, demo_context = choose_demo_pair(data["filtered_tokenized"], data["word2idx"])
    center_idx = data["word2idx"][demo_center]
    context_idx = data["word2idx"][demo_context]

    verify_model = SkipGramNegativeSampling(len(data["vocab"]), embed_dim=embed_dim, seed=seed)
    verify_rng = np.random.RandomState(seed)
    demo_negatives = verify_model.sample_negative_indices(context_idx, num_negative, distribution, verify_rng)
    verify_update = verify_model.update_pair(center_idx, context_idx, demo_negatives, lr_init)

    train_model = SkipGramNegativeSampling(len(data["vocab"]), embed_dim=embed_dim, seed=seed)
    losses = train_negative_sampling(
        train_model,
        data["pairs"],
        distribution,
        epochs=epochs,
        lr_init=lr_init,
        lr_decay=lr_decay,
        num_negative=num_negative,
        shuffle_seed=seed,
    )

    if output_dir is not None:
        plot_losses(losses, output_dir / "loss_curve_neg_sampling.png", "Negative Sampling Loss Curve")

    return {
        **data,
        "distribution": distribution,
        "demo_center": demo_center,
        "demo_context": demo_context,
        "demo_negatives": demo_negatives,
        "verification": verify_update,
        "model": train_model,
        "losses": losses,
        "motivation": (
            "Negative Sampling replaces the expensive full softmax normalization with a binary objective "
            "that updates one positive context word and a small number of sampled negative words."
        ),
        "settings": {
            "min_count": min_count,
            "embed_dim": embed_dim,
            "window_size": window_size,
            "epochs": epochs,
            "lr_init": lr_init,
            "lr_decay": lr_decay,
            "num_negative": num_negative,
            "seed": seed,
        },
    }
