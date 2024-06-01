import numpy as np


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


def diversity_sampling(estimator, X, n, omit_idx=None):
    diversity = np.mean(estimator.predict_proba(X), axis=1)
    valid_indices, mask = get_valid_idx(len(X), omit_idx)

    valid_diversity = diversity[mask]
    n = min(len(valid_diversity), n)

    idx = valid_indices[np.argpartition(valid_diversity, n - 1)[-n:]]
    return idx, X[idx]


def random_sampling(X, n, omit_idx=None):
    valid_indices, _ = get_valid_idx(len(X), omit_idx)
    idx = np.random.choice(valid_indices, n, replace=False)

    return idx, X[idx]
