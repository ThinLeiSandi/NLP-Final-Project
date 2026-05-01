from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.txt"


def load_corpus(corpus_path: Path, max_sentences: int | None = None) -> list[str]:
    lines = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_sentences is not None:
        lines = lines[:max_sentences]
    return lines


def tokenize_corpus(corpus: Iterable[str]) -> list[list[str]]:
    return [sentence.lower().split() for sentence in corpus]


def build_vocab(
    tokenized_corpus: Iterable[Iterable[str]],
    min_count: int = 1,
) -> tuple[list[str], dict[str, int], dict[int, str], dict[str, int]]:
    counts: dict[str, int] = {}
    for sentence in tokenized_corpus:
        for token in sentence:
            counts[token] = counts.get(token, 0) + 1

    vocab = sorted([word for word, count in counts.items() if count >= min_count])
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    filtered_counts = {word: counts[word] for word in vocab}
    return vocab, word2idx, idx2word, filtered_counts


def filter_tokenized_corpus(
    tokenized_corpus: Iterable[Iterable[str]],
    word2idx: dict[str, int],
) -> list[list[str]]:
    filtered: list[list[str]] = []
    for sentence in tokenized_corpus:
        kept = [word for word in sentence if word in word2idx]
        if len(kept) >= 2:
            filtered.append(kept)
    return filtered


def generate_pairs(
    tokenized_corpus: Iterable[Iterable[str]],
    word2idx: dict[str, int],
    window_size: int = 2,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        for center_pos, center_idx in enumerate(indices):
            left = max(0, center_pos - window_size)
            right = min(len(indices), center_pos + window_size + 1)
            for context_pos in range(left, right):
                if context_pos == center_pos:
                    continue
                pairs.append((center_idx, indices[context_pos]))
    return pairs


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def cross_entropy_loss(y_hat: np.ndarray, target_idx: int) -> float:
    return float(-np.log(y_hat[target_idx] + 1e-12))


def format_array(values: np.ndarray, precision: int = 4) -> str:
    return np.array2string(values, precision=precision, suppress_small=False)


def plot_losses(losses: list[float], output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(losses) + 1), losses, color="navy", lw=1.6, marker="o", ms=3)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def choose_demo_pair(
    tokenized_corpus: list[list[str]],
    word2idx: dict[str, int],
) -> tuple[str, str]:
    for sentence in tokenized_corpus:
        if len(sentence) >= 2:
            center_pos = min(1, len(sentence) - 1)
            context_pos = min(center_pos + 1, len(sentence) - 1)
            if context_pos != center_pos:
                return sentence[center_pos], sentence[context_pos]
    vocab_words = sorted(word2idx.keys())
    return vocab_words[0], vocab_words[1]


def bottleneck_explanation(vocab_size: int, embed_dim: int) -> str:
    return (
        f"Full softmax is expensive because every update scores all {vocab_size} vocabulary words, "
        f"so each pair costs O(Vd) = O({vocab_size} x {embed_dim}). "
        "As vocabulary grows, both normalization and gradient computation become much slower."
    )


@dataclass
class GradientCheckResult:
    matrix: str
    index: tuple[int, int]
    analytical: float
    numerical: float
    relative_error: float


class SkipGramFullSoftmax:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0, init_scale: float = 0.01):
        np.random.seed(seed)
        self.V = vocab_size
        self.d = embed_dim
        self.W_in = np.random.randn(vocab_size, embed_dim) * init_scale
        self.W_out = np.random.randn(embed_dim, vocab_size) * init_scale

    def forward(self, center_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v_c = self.W_in[center_idx].copy()
        scores = self.W_out.T @ v_c
        y_hat = softmax(scores)
        return v_c, scores, y_hat

    def backward(
        self,
        center_idx: int,
        context_idx: int,
        v_c: np.ndarray,
        y_hat: np.ndarray,
        lr: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = y_hat.copy()
        error[context_idx] -= 1.0
        grad_W_out = np.outer(v_c, error)
        grad_v_c = self.W_out @ error
        self.W_out -= lr * grad_W_out
        self.W_in[center_idx] -= lr * grad_v_c
        return error, grad_W_out, grad_v_c

    def loss_for_pair(self, center_idx: int, context_idx: int) -> float:
        _, _, y_hat = self.forward(center_idx)
        return cross_entropy_loss(y_hat, context_idx)

    def analytical_gradients(self, center_idx: int, context_idx: int) -> tuple[np.ndarray, np.ndarray]:
        v_c, _, y_hat = self.forward(center_idx)
        error = y_hat.copy()
        error[context_idx] -= 1.0
        grad_W_out = np.outer(v_c, error)
        grad_W_in = np.zeros_like(self.W_in)
        grad_W_in[center_idx] = self.W_out @ error
        return grad_W_in, grad_W_out


def numerical_gradient_for_entry(
    model: SkipGramFullSoftmax,
    matrix_name: str,
    index: tuple[int, int],
    center_idx: int,
    context_idx: int,
    eps: float,
) -> float:
    matrix = getattr(model, matrix_name)
    original_value = matrix[index]
    matrix[index] = original_value + eps
    loss_plus = model.loss_for_pair(center_idx, context_idx)
    matrix[index] = original_value - eps
    loss_minus = model.loss_for_pair(center_idx, context_idx)
    matrix[index] = original_value
    return float((loss_plus - loss_minus) / (2 * eps))


def test_gradients(
    model: SkipGramFullSoftmax,
    center_idx: int,
    context_idx: int,
    eps: float = 1e-5,
    num_checks: int = 5,
    seed: int = 123,
) -> tuple[bool, list[GradientCheckResult]]:
    rng = np.random.RandomState(seed)
    grad_W_in, grad_W_out = model.analytical_gradients(center_idx, context_idx)
    results: list[GradientCheckResult] = []

    in_columns = rng.choice(model.d, size=min(num_checks, model.d), replace=False)
    for col in in_columns:
        index = (center_idx, int(col))
        analytical = float(grad_W_in[index])
        numerical = numerical_gradient_for_entry(model, "W_in", index, center_idx, context_idx, eps)
        relative_error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-12)
        results.append(GradientCheckResult("W_in", index, analytical, numerical, relative_error))

    checked: set[tuple[int, int]] = set()
    while len(checked) < num_checks:
        index = tuple(int(rng.randint(dim)) for dim in grad_W_out.shape)
        if index in checked:
            continue
        checked.add(index)
        analytical = float(grad_W_out[index])
        numerical = numerical_gradient_for_entry(model, "W_out", index, center_idx, context_idx, eps)
        relative_error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-12)
        results.append(GradientCheckResult("W_out", index, analytical, numerical, relative_error))

    return all(result.relative_error < 1e-5 for result in results), results


def train_full_softmax(
    model: SkipGramFullSoftmax,
    pairs: list[tuple[int, int]],
    epochs: int,
    lr_init: float,
    lr_decay: float,
    shuffle_seed: int = 0,
) -> list[float]:
    losses: list[float] = []
    rng = np.random.RandomState(shuffle_seed)
    for epoch in range(1, epochs + 1):
        lr = lr_init / (1.0 + lr_decay * epoch)
        shuffled_indices = rng.permutation(len(pairs))
        total_loss = 0.0
        for pair_idx in shuffled_indices:
            center_idx, context_idx = pairs[pair_idx]
            v_c, _scores, y_hat = model.forward(center_idx)
            total_loss += cross_entropy_loss(y_hat, context_idx)
            model.backward(center_idx, context_idx, v_c, y_hat, lr)
        losses.append(total_loss / len(pairs))
    return losses


def prepare_dataset(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    max_sentences: int | None = None,
    min_count: int = 1,
    window_size: int = 2,
) -> dict[str, object]:
    corpus = load_corpus(corpus_path, max_sentences=max_sentences)
    tokenized = tokenize_corpus(corpus)
    vocab, word2idx, idx2word, counts = build_vocab(tokenized, min_count=min_count)
    filtered_tokenized = filter_tokenized_corpus(tokenized, word2idx)
    pairs = generate_pairs(filtered_tokenized, word2idx, window_size=window_size)
    return {
        "corpus": corpus,
        "tokenized": tokenized,
        "filtered_tokenized": filtered_tokenized,
        "vocab": vocab,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "counts": counts,
        "pairs": pairs,
    }


def run_baseline_experiment(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_dir: Path | None = None,
    subset_sentences: int = 2000,
    min_count: int = 1,
    embed_dim: int = 50,
    window_size: int = 2,
    epochs: int = 10,
    lr_init: float = 0.025,
    lr_decay: float = 0.005,
    seed: int = 0,
) -> dict[str, object]:
    data = prepare_dataset(
        corpus_path=corpus_path,
        max_sentences=subset_sentences,
        min_count=min_count,
        window_size=window_size,
    )
    demo_center, demo_context = choose_demo_pair(data["filtered_tokenized"], data["word2idx"])
    center_idx = data["word2idx"][demo_center]
    context_idx = data["word2idx"][demo_context]

    verify_model = SkipGramFullSoftmax(len(data["vocab"]), embed_dim=embed_dim, seed=seed)
    demo_v_c, demo_scores, demo_y_hat = verify_model.forward(center_idx)
    verify_error, verify_grad_w_out, verify_grad_v_c = verify_model.backward(
        center_idx, context_idx, demo_v_c, demo_y_hat, lr=lr_init
    )

    gradient_model = SkipGramFullSoftmax(len(data["vocab"]), embed_dim=embed_dim, seed=seed)
    gradient_ok, gradient_results = test_gradients(gradient_model, center_idx, context_idx)

    train_model = SkipGramFullSoftmax(len(data["vocab"]), embed_dim=embed_dim, seed=seed)
    losses = train_full_softmax(
        train_model,
        data["pairs"],
        epochs=epochs,
        lr_init=lr_init,
        lr_decay=lr_decay,
        shuffle_seed=seed,
    )

    if output_dir is not None:
        plot_losses(losses, output_dir / "loss_curve_baseline.png", "Baseline Skip-gram Loss Curve")

    return {
        **data,
        "demo_center": demo_center,
        "demo_context": demo_context,
        "model": train_model,
        "losses": losses,
        "verification": {
            "v_c": demo_v_c,
            "scores": demo_scores,
            "y_hat": demo_y_hat,
            "loss": cross_entropy_loss(demo_y_hat, context_idx),
            "error": verify_error,
            "grad_v_c": verify_grad_v_c,
            "grad_w_out_context": verify_grad_w_out[:, context_idx].copy(),
            "updated_center": verify_model.W_in[center_idx].copy(),
            "updated_out_context": verify_model.W_out[:, context_idx].copy(),
        },
        "gradient_check": {
            "passed": gradient_ok,
            "results": gradient_results,
        },
        "bottleneck": bottleneck_explanation(len(data["vocab"]), embed_dim),
        "settings": {
            "subset_sentences": subset_sentences,
            "min_count": min_count,
            "embed_dim": embed_dim,
            "window_size": window_size,
            "epochs": epochs,
            "lr_init": lr_init,
            "lr_decay": lr_decay,
            "seed": seed,
        },
    }
