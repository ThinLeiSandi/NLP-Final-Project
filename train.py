from __future__ import annotations

from pathlib import Path

from baseline_skipgram import format_array, run_baseline_experiment
from negative_sampling import run_negative_sampling_experiment


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_baseline_report(results: dict[str, object]) -> str:
    counts = sorted(results["counts"].items(), key=lambda item: item[1], reverse=True)[:10]
    verification = results["verification"]
    gradient_check = results["gradient_check"]
    losses = results["losses"]
    settings = results["settings"]
    token_count = sum(len(sentence) for sentence in results["filtered_tokenized"])

    lines = [
        "Part A - Baseline Skip-gram from Scratch",
        "A1. Corpus preprocessing and pair generation",
        f"Reduced subset sentences: {settings['subset_sentences']}",
        f"Filtered token count: {token_count}",
        f"Vocabulary size (min_count={settings['min_count']}): {len(results['vocab'])}",
        f"Training pair count (window={settings['window_size']}): {len(results['pairs'])}",
        f"Most frequent words: {counts}",
        "",
        "A2. Baseline model implementation check",
        f"Demo pair: center='{results['demo_center']}', context='{results['demo_context']}'",
        f"v_center = {format_array(verification['v_c'])}",
        f"score(context) = {verification['scores'][results['word2idx'][results['demo_context']]]:.6f}",
        f"P(context | center) = {verification['y_hat'][results['word2idx'][results['demo_context']]]:.6f}",
        f"loss = {verification['loss']:.6f}",
        f"error[context] = {verification['error'][results['word2idx'][results['demo_context']]]:.6f}",
        f"grad_v_center = {format_array(verification['grad_v_c'])}",
        f"grad_W_out[:, context] = {format_array(verification['grad_w_out_context'])}",
        "",
        "A3. Correctness verification",
        f"Gradient check passed = {gradient_check['passed']}",
    ]
    for result in gradient_check["results"]:
        lines.append(
            f"{result.matrix}{result.index}: analytical={result.analytical:.8f}, "
            f"numerical={result.numerical:.8f}, rel_error={result.relative_error:.3e}"
        )

    lines.extend(
        [
            "",
            "A4. Baseline training and loss curve",
            f"Hyperparameters: d={settings['embed_dim']}, window={settings['window_size']}, epochs={settings['epochs']}",
            f"epoch 1: avg_loss={losses[0]:.6f}",
            f"epoch {max(1, settings['epochs'] // 2)}: avg_loss={losses[max(1, settings['epochs'] // 2) - 1]:.6f}",
            f"epoch {settings['epochs']}: avg_loss={losses[-1]:.6f}",
            "Saved loss curve: loss_curve_baseline.png",
            "",
            "A5. Full-softmax bottleneck explanation",
            results["bottleneck"],
            "",
        ]
    )
    return "\n".join(lines)


def build_negative_sampling_report(neg_results: dict[str, object], baseline_results: dict[str, object]) -> str:
    verification = neg_results["verification"]
    losses = neg_results["losses"]
    settings = neg_results["settings"]
    token_count = sum(len(sentence) for sentence in neg_results["filtered_tokenized"])
    baseline_losses = baseline_results["losses"]

    lines = [
        "Part B - Negative Sampling Extension",
        "B1. Motivation and formulation",
        neg_results["motivation"],
        "",
        "B2. Negative Sampling implementation",
        f"Demo pair: center='{neg_results['demo_center']}', context='{neg_results['demo_context']}'",
        f"Sampled negative indices: {neg_results['demo_negatives']}",
        f"Sampled negative words: {[neg_results['idx2word'][idx] for idx in neg_results['demo_negatives']]}",
        f"one-step loss = {verification['loss']:.6f}",
        f"grad_W_in[center] = {format_array(verification['grad_W_in'])}",
        f"grad_W_out[:, context] = {format_array(verification['grad_W_out_context'])}",
        "",
        "B3. Training on the full standardized corpus",
        f"Full-corpus sentences: {len(neg_results['corpus'])}",
        f"Full-corpus token count: {token_count}",
        f"Vocabulary size (min_count={settings['min_count']}): {len(neg_results['vocab'])}",
        f"Training pair count (window={settings['window_size']}): {len(neg_results['pairs'])}",
        f"Negative samples per positive pair: {settings['num_negative']}",
        f"epoch 1: avg_loss={losses[0]:.6f}",
        f"epoch {max(1, settings['epochs'] // 2)}: avg_loss={losses[max(1, settings['epochs'] // 2) - 1]:.6f}",
        f"epoch {settings['epochs']}: avg_loss={losses[-1]:.6f}",
        "Saved loss curve: loss_curve_neg_sampling.png",
        "",
        "B4. Comparison with baseline training behaviour",
        f"Baseline final loss on reduced subset = {baseline_losses[-1]:.6f}",
        f"Negative Sampling final loss on full corpus = {losses[-1]:.6f}",
        "Negative Sampling scales better because each update touches only one positive context word and a small set of sampled negatives, rather than the whole vocabulary.",
        "This makes full-corpus training practical while preserving the same center-context learning idea.",
        "",
        "B5. Technical clarity",
        "Compared with the baseline, the extension replaces the full-softmax objective with a binary positive-vs-negative objective and adds stochastic negative example generation from a unigram^0.75 distribution.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_results = run_baseline_experiment(
        corpus_path=PROJECT_DIR / "data" / "corpus.txt",
        output_dir=PROJECT_DIR,
        subset_sentences=2000,
        min_count=1,
        embed_dim=50,
        window_size=2,
        epochs=5,
        lr_init=0.025,
        lr_decay=0.005,
        seed=0,
    )
    neg_results = run_negative_sampling_experiment(
        corpus_path=PROJECT_DIR / "data" / "corpus.txt",
        output_dir=PROJECT_DIR,
        min_count=1,
        embed_dim=50,
        window_size=2,
        epochs=5,
        lr_init=0.025,
        lr_decay=0.0,
        num_negative=5,
        seed=0,
    )

    baseline_report = build_baseline_report(baseline_results)
    neg_report = build_negative_sampling_report(neg_results, baseline_results)
    write_text(RESULTS_DIR / "baseline_output.txt", baseline_report)
    write_text(RESULTS_DIR / "negative_sampling_output.txt", neg_report)

    print(baseline_report)
    print(neg_report)


if __name__ == "__main__":
    main()
