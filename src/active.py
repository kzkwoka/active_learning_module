import numpy as np

from strategies import uncertainty_sampling
class ActiveModule:
    def __init__(self,
                 estimator,
                 strategy=uncertainty_sampling,
                 X_train=None,
                 y_train=None,
                 X_pool=None):
        self.estimator = estimator
        self.strategy = strategy

        self.X_train = X_train
        self.y_train = y_train

        self.X_pool = X_pool
        self.used_idx = np.array([],dtype='int')

    def fit(self, X_train=None, y_train=None, q_idx=None):
        self.X_train = X_train if X_train is not None else self.X_train
        self.y_train = y_train if y_train is not None else self.y_train
        if q_idx is not None and X_train is not None and y_train is not None:
            self.used_idx = np.concatenate((self.used_idx, q_idx))
            self.X_train = np.vstack((self.X_train, X_train))
            self.y_train = np.vstack((self.y_train.reshape(-1,1), y_train.reshape(-1,1))).shape
        self.estimator.fit(X_train, y_train)

    def query(self, X_pool=None, strategy=None, n_samples=None):
        self.strategy = strategy if strategy else self.strategy
        self.X_pool = X_pool if X_pool is not None else self.X_pool

        if self.X_pool is None:
            raise TypeError(f"X_pool should be type sth got {type(self.X_pool)}")# TODO: choose type
        if not callable(self.strategy):
            raise TypeError(f"strategy should be callable got {type(self.strategy)}")

        n_samples = n_samples if n_samples else int(0.1*len(X_pool))
        return self.strategy(self.estimator, self.X_pool, n_samples, omit_idx=self.used_idx)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X):
        return self.estimator.score(X)


