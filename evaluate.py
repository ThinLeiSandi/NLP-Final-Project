from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from scipy.stats import spearmanr

from baseline_skipgram import DEFAULT_CORPUS_PATH, prepare_dataset, run_baseline_experiment
from negative_sampling import run_negative_sampling_experiment


DEFAULT_EVAL_WORD_PAIRS = [
    ("cat", "dog"),
    ("cat", "rabbit"),
    ("dog", "fox"),
    ("teacher", "student"),
    ("doctor", "nurse"),
    ("king", "queen"),
    ("man", "woman"),
    ("language", "model"),
    ("token", "embedding"),
    ("city", "village"),
    ("river", "sea"),
    ("book", "library"),
]

DEFAULT_QUERY_WORDS = [
    "cat",
    "dog",
    "teacher",
    "student",
    "doctor",
    "nurse",
    "king",
    "queen",
    "language",
    "model",
    "embedding",
    "city",
    "river",
]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def top_k_neighbors(
    word: str,
    embeddings: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    k: int = 3,
) -> list[tuple[str, float]]:
    query_idx = word2idx[word]
    query_vec = embeddings[query_idx]
    scores: list[tuple[str, float]] = []
    for idx, candidate in idx2word.items():
        if idx == query_idx:
            continue
        scores.append((candidate, cosine_similarity(query_vec, embeddings[idx])))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:k]


def pair_interpretation(word_a: str, word_b: str, score: float) -> str:
    if score >= 0.75:
        strength = "very strong"
        strength_comment = "embeddings are very close in vector space, indicating highly overlapping contexts"
    elif score >= 0.55:
        strength = "strong"
        strength_comment = "embeddings are close, reflecting consistent co-occurrence patterns"
    elif score >= 0.30:
        strength = "moderate"
        strength_comment = "embeddings share some context overlap but are not tightly clustered"
    elif score >= 0.00:
        strength = "weak"
        strength_comment = "little shared context was captured during training"
    else:
        strength = "negative"
        strength_comment = "embeddings point in opposite directions, suggesting contrasting contexts"

    pair_set = frozenset({word_a, word_b})
    if pair_set in (frozenset({"teacher", "student"}), frozenset({"doctor", "nurse"})):
        category = "profession/role pair; similarity reflects shared occupational sentence contexts"
    elif pair_set == frozenset({"king", "queen"}):
        category = "relational pair; similarity driven by repeated relational sentence templates"
    elif pair_set == frozenset({"man", "woman"}):
        category = "gender-relational pair; similarity shaped by symmetric sentence frames"
    elif pair_set in (frozenset({"language", "model"}), frozenset({"token", "embedding"})):
        category = "technical pair; similarity reflects repeated instructional co-occurrence patterns"
    elif pair_set in (frozenset({"city", "village"}), frozenset({"river", "sea"})):
        category = "geographic/topical pair; similarity reflects shared environmental sentence contexts"
    elif pair_set == frozenset({"book", "library"}):
        category = "cultural pair; similarity reflects repeated sentence-level associations"
    elif pair_set in (frozenset({"cat", "dog"}), frozenset({"cat", "rabbit"}), frozenset({"dog", "fox"})):
        category = "animal pair; similarity reflects shared template positions in the corpus"
    else:
        category = "general pair; similarity reflects local co-occurrence patterns"

    return f"{strength} similarity (score={score:.4f}); {strength_comment}; {category}"


def neighbor_observation(word: str, neighbors: list[tuple[str, float]]) -> str:
    # Interpretation derived from actual neighbor names and scores, not fixed strings
    neighbor_words = [candidate for candidate, _score in neighbors]
    neighbor_scores = [score for _candidate, score in neighbors]
    joined = ", ".join(f"{w} ({s:.4f})" for w, s in neighbors)

    avg_score = sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0
    top_score = neighbor_scores[0] if neighbor_scores else 0.0

    if avg_score >= 0.75:
        cohesion = "very tight cluster (avg neighbor similarity >= 0.75)"
    elif avg_score >= 0.55:
        cohesion = "reasonably tight cluster (avg neighbor similarity >= 0.55)"
    elif avg_score >= 0.30:
        cohesion = "loose cluster (avg neighbor similarity >= 0.30)"
    else:
        cohesion = "very diffuse neighborhood (avg neighbor similarity < 0.30)"

    # Check whether top neighbor is semantically plausible for this word
    animal_words = {"cat", "dog", "fox", "rabbit", "tiger", "lion"}
    profession_words = {"teacher", "student", "doctor", "nurse", "engineer", "scientist", "manager"}
    technical_words = {"language", "model", "embedding", "token", "algorithm", "python", "vector"}

    if word in animal_words:
        domain = "animal"
        plausible = animal_words
    elif word in profession_words:
        domain = "profession/role"
        plausible = profession_words
    elif word in technical_words:
        domain = "technical/NLP"
        plausible = technical_words
    else:
        domain = "general"
        plausible = set()

    plausible_neighbors = [w for w in neighbor_words if w in plausible]
    surprising_neighbors = [w for w in neighbor_words if w not in plausible and plausible]

    if plausible_neighbors and surprising_neighbors:
        semantic_comment = (
            f"plausible {domain} neighbors: {', '.join(plausible_neighbors)}; "
            f"surprising neighbors: {', '.join(surprising_neighbors)} (likely corpus template artifacts)"
        )
    elif plausible_neighbors:
        semantic_comment = f"all neighbors are plausible {domain} words, suggesting a clean semantic cluster"
    elif plausible:
        semantic_comment = f"no clearly plausible {domain} neighbors found; results may reflect corpus frequency effects"
    else:
        semantic_comment = "neighbors reflect general co-occurrence patterns from the corpus"

    return (
        f"Top-3 neighbors: {joined}. "
        f"Neighborhood cohesion: {cohesion}. "
        f"Semantic quality: {semantic_comment}."
    )


def evaluate_embeddings(
    model_name: str,
    embeddings: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    word_pairs: list[tuple[str, str]],
    query_words: list[str],
    k: int = 3,
) -> dict[str, object]:
    similarities = []
    for word_a, word_b in word_pairs:
        if word_a not in word2idx or word_b not in word2idx:
            continue
        score = cosine_similarity(embeddings[word2idx[word_a]], embeddings[word2idx[word_b]])
        similarities.append(
            {
                "pair": (word_a, word_b),
                "score": score,
                "interpretation": pair_interpretation(word_a, word_b, score),
            }
        )
    similarities.sort(key=lambda item: item["score"], reverse=True)

    neighbors = []
    for word in query_words:
        if word not in word2idx:
            continue
        top3 = top_k_neighbors(word, embeddings, word2idx, idx2word, k=k)
        neighbors.append(
            {
                "query": word,
                "neighbors": top3,
                "observation": neighbor_observation(word, top3),
            }
        )

    return {
        "model_name": model_name,
        "similarities": similarities,
        "neighbors": neighbors,
    }


def train_gensim_model(
    tokenized_corpus: list[list[str]],
    embed_dim: int,
    window_size: int,
    epochs: int,
    seed: int,
    min_count: int = 1,
    negative: int = 5,
) -> Word2Vec:
    return Word2Vec(
        sentences=tokenized_corpus,
        vector_size=embed_dim,
        window=window_size,
        sg=1,
        min_count=min_count,
        workers=1,
        seed=seed,
        epochs=epochs,
        negative=negative,
        sample=0.0,
    )


def gensim_neighbors(model: Word2Vec, word: str, k: int = 3) -> list[tuple[str, float]]:
    return [(candidate, float(score)) for candidate, score in model.wv.most_similar(word, topn=k)]


def build_part_c_report(
    baseline_eval: dict[str, object],
    neg_eval: dict[str, object],
    baseline_results: dict[str, object],
    neg_results: dict[str, object],
) -> str:
    lines = [
        "Part C - Intrinsic Embedding Evaluation",
        "C1. Cosine similarity evaluation",
        "",
        "Baseline model similarities (reduced subset, full softmax):",
    ]
    for item in baseline_eval["similarities"]:
        lines.append(f"{item['pair']}: {item['score']:.4f} -> {item['interpretation']}")

    lines.extend(["", "Negative Sampling model similarities (full corpus):"])
    for item in neg_eval["similarities"]:
        lines.append(f"{item['pair']}: {item['score']:.4f} -> {item['interpretation']}")

    lines.extend(["", "C2. Nearest-neighbour retrieval", "", "Baseline model neighbors:"])
    for item in baseline_eval["neighbors"]:
        neighbor_text = ", ".join(f"{word} ({score:.4f})" for word, score in item["neighbors"])
        lines.append(f"{item['query']}: {neighbor_text}")
        lines.append(f"  Observation: {item['observation']}")

    lines.extend(["", "Negative Sampling model neighbors:"])
    for item in neg_eval["neighbors"]:
        neighbor_text = ", ".join(f"{word} ({score:.4f})" for word, score in item["neighbors"])
        lines.append(f"{item['query']}: {neighbor_text}")
        lines.append(f"  Observation: {item['observation']}")

    lines.extend(
        [
            "",
            "C3. Quality of interpretation",
            "The corpus is moderately sized for a classroom project, but it still uses many repeated sentence templates. "
            "This means embeddings are strongly shaped by local co-occurrence patterns and frequency effects.",
            "High-frequency template words can pull multiple concepts closer together than expected, which is why some neighbors are meaningful while others are noisy.",
            "The Negative Sampling model is trained on the full corpus, so it usually gives more stable similarities and neighbors than the baseline full-softmax model trained on a reduced subset.",
            "Corpus bias also matters: semantic relations that are repeated often, such as profession pairs or technical terms, are easier to learn than rare or weakly connected words.",
            "",
            f"Baseline corpus size used for Part C: {len(baseline_results['corpus'])} sentences, {sum(len(s) for s in baseline_results['filtered_tokenized'])} filtered tokens.",
            f"Negative Sampling corpus size used for Part C: {len(neg_results['corpus'])} sentences, {sum(len(s) for s in neg_results['filtered_tokenized'])} filtered tokens.",
            "",
        ]
    )
    return "\n".join(lines)


def build_part_d_report(
    neg_eval: dict[str, object],
    gensim_model: Word2Vec,
    word_pairs: list[tuple[str, str]],
    query_words: list[str],
) -> str:
    comparison_rows = []
    custom_scores = []
    gensim_scores = []

    neg_similarity_map = {item["pair"]: item["score"] for item in neg_eval["similarities"]}
    for pair in word_pairs:
        word_a, word_b = pair
        if word_a not in gensim_model.wv.key_to_index or word_b not in gensim_model.wv.key_to_index:
            continue
        if pair not in neg_similarity_map:
            continue
        custom_score = float(neg_similarity_map[pair])
        gensim_score = float(gensim_model.wv.similarity(word_a, word_b))
        custom_scores.append(custom_score)
        gensim_scores.append(gensim_score)
        comparison_rows.append({"pair": pair, "custom": custom_score, "gensim": gensim_score})

    rho, p_value = spearmanr(custom_scores, gensim_scores)
    custom_ranked = sorted(comparison_rows, key=lambda item: item["custom"], reverse=True)
    gensim_ranked = sorted(comparison_rows, key=lambda item: item["gensim"], reverse=True)
    custom_ranks = {item["pair"]: rank for rank, item in enumerate(custom_ranked, start=1)}
    gensim_ranks = {item["pair"]: rank for rank, item in enumerate(gensim_ranked, start=1)}

    lines = [
        "Part D - Comparison with Gensim",
        "D1. Experimental comparison",
        "The Gensim model was trained on the same full standardized corpus with closely matched visible settings: sg=1, vector_size=50, window=2, min_count=1, epochs=5, workers=1, seed=0, negative=5, sample=0.",
        "",
        "Similarity comparison (Negative Sampling model vs Gensim):",
    ]
    for row in comparison_rows:
        lines.append(
            f"{row['pair']}: custom={row['custom']:.4f}, gensim={row['gensim']:.4f}, "
            f"custom_rank={custom_ranks[row['pair']]}, gensim_rank={gensim_ranks[row['pair']]}"
        )

    lines.extend(["", "Nearest-neighbour comparison:"])
    for word in query_words:
        if word not in gensim_model.wv.key_to_index:
            continue
        custom_neighbors = next((item["neighbors"] for item in neg_eval["neighbors"] if item["query"] == word), None)
        if custom_neighbors is None:
            continue
        gensim_top = gensim_neighbors(gensim_model, word, k=3)
        lines.append(f"{word}:")
        lines.append("  custom -> " + ", ".join(f"{w} ({s:.4f})" for w, s in custom_neighbors))
        lines.append("  gensim -> " + ", ".join(f"{w} ({s:.4f})" for w, s in gensim_top))

    lines.extend(
        [
            "",
            "D2. Result analysis",
            f"Spearman rank correlation between similarity lists: rho = {float(rho):.4f}, p-value = {float(p_value):.4f}",
            "The comparison should focus more on whether the models preserve similar relative structure than on exact numerical equality.",
            "Differences are expected for at least four reasons:",
            "1. Gensim uses its own optimized implementation and update ordering, so training dynamics differ even with the same seed.",
            "2. Initialization details can differ slightly, which changes the trajectory of stochastic training.",
            "3. Training schedule details such as internal batching and iteration order can produce different local optima.",
            "4. Frequency effects and implementation-level optimizations in Gensim can reshape neighborhood structure even when the visible hyperparameters match.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part C and D evaluation for Word2Vec final project.")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = run_baseline_experiment(
        corpus_path=args.corpus_path,
        output_dir=None,
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
        corpus_path=args.corpus_path,
        output_dir=None,
        min_count=1,
        embed_dim=50,
        window_size=2,
        epochs=5,
        lr_init=0.025,
        lr_decay=0.0,
        num_negative=5,
        seed=0,
    )

    word_pairs = DEFAULT_EVAL_WORD_PAIRS
    query_words = DEFAULT_QUERY_WORDS

    baseline_eval = evaluate_embeddings(
        "baseline_full_softmax",
        baseline_results["model"].W_in,
        baseline_results["word2idx"],
        baseline_results["idx2word"],
        word_pairs,
        query_words,
    )
    neg_eval = evaluate_embeddings(
        "negative_sampling",
        neg_results["model"].W_in,
        neg_results["word2idx"],
        neg_results["idx2word"],
        word_pairs,
        query_words,
    )

    full_data = prepare_dataset(corpus_path=args.corpus_path, max_sentences=None, min_count=1, window_size=2)
    gensim_model = train_gensim_model(
        tokenized_corpus=full_data["filtered_tokenized"],
        embed_dim=50,
        window_size=2,
        epochs=5,
        seed=0,
        min_count=1,
        negative=5,
    )

    part_c_report = build_part_c_report(baseline_eval, neg_eval, baseline_results, neg_results)
    part_d_report = build_part_d_report(neg_eval, gensim_model, word_pairs, query_words)
    full_report = part_c_report + "\n" + part_d_report

    write_text(args.results_dir / "part_c_output.txt", part_c_report)
    write_text(args.results_dir / "part_d_output.txt", part_d_report)
    write_text(args.results_dir / "evaluation_output.txt", full_report)
    print(full_report)


if __name__ == "__main__":
    main()