from typing import Type
from warnings import warn
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .strategies import uncertainty_sampling, random_sampling
except ImportError:
    from strategies import uncertainty_sampling, random_sampling


class SklearnLikeModel(BaseEstimator):
    def __init__(self, **kwargs):
        ...

    def fit(self, X, y):
        ...

    def score(self, X, y):
        ...

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...


# noinspection PyProtectedMember
def _get_pred_matrix(y, y_pred, num_classes):
    matrix = np.zeros((y.shape[0], num_classes))
    matrix[np.arange(y.shape[0]), y] = 1
    matrix_pred = np.zeros_like(matrix)
    matrix_pred[np.arange(y_pred.shape[0]), y_pred] = 1
    return matrix, matrix_pred


class ActiveModule:
    def __init__(self,
                 estimator: Type[SklearnLikeModel],
                 X: np._typing.ArrayLike,
                 y_initial: np._typing.ArrayLike = None,
                 label_idx=np.array([], dtype='int'),
                 strategy=uncertainty_sampling,
                 X_valid: np._typing.ArrayLike = None,
                 y_valid: np._typing.ArrayLike = None,
                 seed: int = 42,
                 **kwargs):
        """
        Active learning module for sckit-learn like models.
        :param estimator: Model to be fitted. Must have `fit`, `predict`, `predict_proba` and `score` methods.
        :param X: Data to be used for fitting the estimator (both labeled and unlabeled).
        :param y_initial: Labels known before fitting the estimator.
        :param label_idx: Indexes of data points with known labels.
        :param strategy: Method of querying for data point with unknown labels.
        :param X_valid: Validation data to be used for scoring the estimator.
        :param y_valid: Validation labels to be used for scoring the estimator.
        """
        np.random.seed(seed)
        self.estimator = estimator(**kwargs)
        self.strategy = strategy

        self.X = X
        self.y = np.empty((len(self.X), 1))
        self.y[:] = np.nan

        self.label_idx = label_idx
        if len(label_idx) > 0 and len(label_idx) == len(y_initial):
            self.X_train = self.X[label_idx]
            self.y_train = self.y[label_idx] = y_initial.reshape(-1, 1)
        else:
            warn(f"`y` should have length {len(label_idx)}, but got {len(y_initial)}. Labels omitted.")

        if X_valid is not None and y_valid is not None:
            self.X_valid = X_valid
            self.y_valid = y_valid

            self._metrics = np.array([], dtype=dict)

            self.classes = np.unique(np.concatenate([self.y_train.flatten(), self.y_valid.flatten()]))
        else:
            self.classes = np.unique(self.y_train)
        self.mapping = {label: idx for idx, label in enumerate(self.classes)}

        self._fitted = False

    def get_metric_history(self):
        return pd.DataFrame(self._metrics.tolist())

    def fit(self, y_train=None, q_idx=None):

        # If passing new labeled data
        if q_idx is not None and y_train is not None:
            self.label_idx = np.concatenate((self.label_idx, q_idx))
            if not hasattr(self, "X_train"):
                self.X_train = self.X[q_idx]
            else:
                self.X_train = np.vstack((self.X_train, self.X[q_idx]))
            self.y[q_idx] = y_train.reshape(-1, 1)
            self.y_train = self.y[self.label_idx]
        elif q_idx is not None or y_train is not None:
            raise ValueError("Neither or both of `q_idx` and `y_train` must be provided.")
        # If training on initial data
        else:
            # Make sure training data available
            if not hasattr(self, "X_train") or not hasattr(self, "y_train"):
                raise ValueError("No training labels found. Both `q_idx` and `y_train` must be provided.")

        self.estimator.fit(X=self.X_train, y=self.y_train.ravel())
        self._fitted = True

    def query(self, strategy=None, n_samples=None):
        strategy = strategy if strategy else self.strategy

        if not callable(strategy):
            raise TypeError(f"strategy should be callable got {type(strategy)}")

        n_samples = n_samples if n_samples else int(0.1 * len(self.X))
        if self._fitted:
            return self.strategy(self.estimator, self.X, n_samples, omit_idx=self.label_idx)
        else:
            warn(f"`estimator` has not been fitted yet. Providing random samples instead.")
            return random_sampling(self.X, n_samples, omit_idx=self.label_idx)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    def get_metrics(self, X, y) -> dict:
        y_pred = self.estimator.predict(X)
        y_mapped = np.array([self.mapping[label] for label in y.flatten()])
        y_pred_mapped = np.array([self.mapping[label] for label in y_pred.flatten().astype(int)])
        matrix, matrix_pred = _get_pred_matrix(y_mapped, y_pred_mapped, len(self.classes))
        return {
            'default_metric': self.score(X, y),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='micro'),
            'recall': recall_score(y, y_pred, average='micro'),
            'f1': f1_score(y, y_pred, average='micro'),
            'roc_auc': roc_auc_score(matrix, matrix_pred),
            'pr_auc': average_precision_score(matrix, matrix_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }

    def step(self, y=None, q_idx=None, strategy=None, n_samples=None):
        self.fit(y, q_idx)
        if hasattr(self, "X_valid") and hasattr(self, "y_valid"):
            self._metrics = np.append(self._metrics, self.get_metrics(self.X_valid, self.y_valid))
        return self.query(strategy, n_samples)
