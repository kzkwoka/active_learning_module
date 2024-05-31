import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from active import ActiveModule
np.random.seed(123)

iris = load_iris()
X_raw = iris['data']
y_raw = iris['target']

# Isolate our examples for our labeled dataset.
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=10)

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)

knn = KNeighborsClassifier(n_neighbors=3)
learner = ActiveModule(estimator=knn, X_train=X_train, y_train=y_train)

learner.fit(X_train, y_train)

learner.predict(X_pool)

q_idx, q_instance = learner.query(X_pool)
X, y = X_pool[q_idx], y_pool[q_idx]

learner.fit(X, y, q_idx=q_idx)


q_idx, _ = learner.query(n_samples=128)
