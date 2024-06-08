import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from active import ActiveModule
import strategies

iris = load_iris()
X_raw = iris['data']
y_raw = iris['target']

initial_idx = np.random.randint(low=0, high=X_raw.shape[0], size=10)
y_initial = y_raw[initial_idx]

learner = ActiveModule(estimator=KNeighborsClassifier,
                       X=X_raw,
                       y_initial=y_initial,
                       label_idx=initial_idx,
                       X_valid=X_raw,
                       y_valid=y_raw,
                       n_neighbors=3)

y, q_idx = None, None
for i in range(14):
    q_idx, _ = learner.step(y, q_idx, strategy=strategies.diversity_sampling, n_samples=10)
    y = y_raw[q_idx]

metrics = learner.get_metric_history()

metrics[["accuracy", "pr_auc", "roc_auc"]].plot()
plt.savefig("kiniaaa.png")
print('hi')