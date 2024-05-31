import numpy as np

# def uncertainty_sampling(estimator, X, n, omit_idx=None):
#     uncertainty = 1 - np.max(estimator.predict_proba(X), axis=1)
#
#     idx = np.argpartition(uncertainty, n, axis=0)[:n]
#     return idx, X[idx]


def uncertainty_sampling(estimator, X, n, omit_idx=None):
    uncertainty = 1 - np.max(estimator.predict_proba(X), axis=1)

    mask = np.ones(len(X), dtype=bool)
    if omit_idx.size != 0:
        mask[omit_idx] = False

    valid_indices = np.where(mask)[0]
    valid_uncertainty = uncertainty[mask]

    n = min(len(valid_uncertainty), n)
    idx = valid_indices[np.argpartition(valid_uncertainty, n-1)[:n]]
    return idx, X[idx]