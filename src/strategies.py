import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import pairwise_distances


def get_valid_idx(length, omit_idx):
    mask = np.ones(length, dtype=bool)
    if omit_idx.size != 0:
        mask[omit_idx] = False

    valid_indices = np.where(mask)[0]
    return valid_indices, mask


# TODO: make sure that the same samples are returned if more have the same uncertainty
def uncertainty_sampling(estimator, X, n, omit_idx=None):
    uncertainty = 1 - np.max(estimator.predict_proba(X), axis=1)

    valid_indices, mask = get_valid_idx(len(X), omit_idx)
    valid_uncertainty = uncertainty[mask]
    n = min(len(valid_uncertainty), n)
    idx = valid_indices[np.argpartition(valid_uncertainty, n - 1)[:n]]
    return idx, X[idx]


def diversity_sampling(estimator, X, n, omit_idx=None, metric='euclidean'):
    valid_indices, mask = get_valid_idx(len(X), omit_idx)
    # Zaczynamy od losowego wybrania pierwszej próbki
    # selected_indices = [np.random.choice(valid_indices)]
    selected_indices = list(omit_idx)
    chosen = []

    for _ in range(n - 1):
        # Obliczamy odległości pomiędzy wybranymi próbkami a wszystkimi próbkami w puli
        distances = pairwise_distances(X[selected_indices], X, metric=metric)

        # Dla każdej próbki w puli obliczamy minimalną odległość do którejkolwiek z wybranych próbek
        min_distances = distances.min(axis=0)

        # Wybieramy próbkę, która ma maksymalną minimalną odległość do wybranych próbek
        next_sample_index = min_distances.argmax()
        chosen.append(next_sample_index)
        selected_indices.append(next_sample_index)

    return chosen, X[chosen]


def kmeans_plus_plus_sampling(estimator, X, n, omit_idx=None):
    valid_indices, mask = get_valid_idx(len(X), omit_idx)
    sub_X = X[valid_indices]
    n = min(len(valid_indices), n)
    centers_init, k_indices = kmeans_plusplus(sub_X, n_clusters=n, random_state=42)
    return valid_indices[k_indices], X[valid_indices[k_indices]]


# def diversity_sampling(estimator, X, n, omit_idx=None):
#     diversity = np.mean(estimator.predict_proba(X), axis=1)
#     valid_indices, mask = get_valid_idx(len(X), omit_idx)
#
#     valid_diversity = diversity[mask]
#     n = min(len(valid_diversity), n)
#
#     idx = valid_indices[np.argpartition(valid_diversity, n - 1)[-n:]]
#     return idx, X[idx]


def random_sampling(estimator, X, n, omit_idx=None):
    valid_indices, _ = get_valid_idx(len(X), omit_idx)
    n = min(len(valid_indices), n)
    idx = np.random.choice(valid_indices, n, replace=False)

    return idx, X[idx]
