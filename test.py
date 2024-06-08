from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from src.active import ActiveModule
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
np.random.seed(42)
initial_idx = np.random.randint(low=0, high=X_train.shape[0], size=int(0.01*X_train.shape[0]))
# initializing the learner
learner = ActiveLearner(
    estimator=KNeighborsClassifier(),
    query_strategy=uncertainty_sampling,
    X_training=X_train[initial_idx], y_training=y_train.values[initial_idx]
)
X_to_train = X_train.copy()
y_to_train = y_train.values.copy()
# query for labels
for i in range(50):
    query_idx, query_inst = learner.query(X_to_train, n_instances=100)
    print(query_idx)
    

    # ...obtaining new labels from the Oracle...

    # supply label for queried instance
    print(query_idx)
    print(X_to_train[query_idx])
    print(y_to_train[query_idx])
    learner.teach(X_to_train[query_idx], y_to_train[query_idx])
    X_to_train = np.delete(X_to_train, query_idx, axis=0)
    y_to_train = np.delete(y_to_train, query_idx, axis=0)
print(learner.score(X_test, y_test))