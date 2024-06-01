## Usage
### Default
Useful when looking to iterate over the whole dataset.
```python
learner = ActiveModule(estimator=...,
                       X=X,
                       y_initial=y_initial,
                       label_idx=initial_idx,
                       X_valid=X_valid,
                       y_valid=y_valid,
                       **estimator_kwargs)

y, q_idx = None, None
for i in range(N):
    q_idx, _ = learner.step(y, q_idx, n_samples=10)
    
    y = ... # Get labels for samples with indices in q_idx

scores = learner.get_scores()
```
### Manual
When needing more control of the training loop
```python
learner = ActiveModule(estimator=...,
                       X=X,
                       y_initial=y_initial,
                       label_idx=initial_idx,
                       **estimator_kwargs)

y, q_idx = None, None
for i in range(N):
    learner.fit(y, q_idx)

    q_idx, _ = learner.query(n_samples=10)
    y = ... # Get labels for samples with indices in q_idx

    score = learner.score(X_valid, y_valid)
```