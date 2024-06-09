from copy import deepcopy

import pandas as pd
from tqdm import tqdm


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def run_experiments(learner, y_train, strategies, n_samples, n_iter):
    metrics = {}
    info = []
    for strategy in tqdm(strategies):
        l = deepcopy(learner)
        _y, q_idx = None, None
        for _ in (t := tqdm(range(n_iter))):
            q_idx, _ = l.step(_y, q_idx, strategy=strategy, n_samples=n_samples)
            _y = y_train[q_idx]
        avg_loop_time = t.format_dict['elapsed'] / t.format_dict['total']
        info.append(f"One training loop with {strategy.__name__} takes approx. {round(avg_loop_time, 2)}s")
        metrics[strategy.__name__] = l.get_metric_history()
    return pd.concat(metrics, names=['strategy']), info
