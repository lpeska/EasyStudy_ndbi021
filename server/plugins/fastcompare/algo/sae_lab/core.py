"""Simple signed-SAE baseline for NDBI021: balanced coverage and explicit feedback."""
import numpy as np

RESPONSES = {'positive', 'negative', 'both', 'neither'}


def select_groups(z, popularity, group_size=4, quantile=0.5, seed=42):
    """Rank neurons by min(#positive movies, #negative movies).

    Require >=group_size active movies on EACH side; sample representatives
    uniformly without replacement. `popularity` and `quantile` are retained
    for compatibility with the previous interface but are not used for ranking.
    The seeded RNG makes study initialization reproducible.
    Return (neuron_id, positive_rows, negative_rows, coverage_score).
    """
    z = np.asarray(z)
    if z.ndim != 2 or len(popularity) != z.shape[0]:
        raise ValueError('Invalid embedding/popularity shape')
    if group_size < 1:
        raise ValueError('group_size must be positive')
    rng = np.random.default_rng(seed)
    groups = []
    for neuron in range(z.shape[1]):
        pos = np.flatnonzero(z[:, neuron] > 0)
        neg = np.flatnonzero(z[:, neuron] < 0)
        if min(len(pos), len(neg)) < group_size:
            continue
        score = min(len(pos), len(neg))
        groups.append((neuron, rng.choice(pos, group_size, replace=False).tolist(),
                       rng.choice(neg, group_size, replace=False).tolist(), score))
    return sorted(groups, key=lambda group: (-group[3], group[0]))


def update_preferences(explicit, implicit, z, dimension, positive, negative, response,
                       alpha=0.3, beta=0.25):
    """Update ONLY explicit signed feature preference; keep implicit zero.

    Both/neither express no preference about the queried sign. Retain the
    previous call signature to avoid changing FastCompare's Flask route.
    """
    if response not in RESPONSES:
        raise ValueError('Unknown feedback')
    explicit = np.asarray(explicit, dtype=np.float32).copy()
    if response == 'positive':
        explicit[dimension] += 1.0
    elif response == 'negative':
        explicit[dimension] -= 1.0
    implicit = np.zeros_like(explicit)
    return explicit, implicit


def recommend(z, explicit, implicit, seen, k, alpha=0.3):
    """Rank by explicit SAE preference alone; filter already shown items."""
    z = np.asarray(z)
    scores = z @ np.asarray(explicit)
    scores = np.asarray(scores, dtype=float)
    if seen:
        scores[np.asarray(list(seen), dtype=int)] = -np.inf
    return np.argsort(-scores, kind='stable')[:min(k, np.isfinite(scores).sum())].tolist()
