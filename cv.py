import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

def run_cv(clf, params, X, y, X_valid, y_valid, n_splits=5, n_repeats=10, threshold=0.3):
    # Initialize RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    results = pd.DataFrame(columns=['clf', 'params', 'accuracy', 'f1', 'precision', 'recall'])

    for i, (train_index, _) in enumerate(rskf.split(X, y), 1):
        # Select training set for this fold
        X_train, y_train = X[train_index], y[train_index]

        # Create a fresh clone of the classifier for this repeat
        clf_clone = clone(clf)

        # Tune hyperparameters on the current fold
        random_search = RandomizedSearchCV(clf_clone, params, n_jobs=-1, cv=5)
        random_search.fit(X_train, y_train)

        # Calculate probabilities on the validation set
        best_clf = random_search.best_estimator_
        probabilities = best_clf.predict_proba(X_valid)
        y_pred = (probabilities[:, 1] >= threshold).astype(int)
        acc = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred,  average='weighted')
        precision = precision_score(y_valid, y_pred,  average='weighted')
        recall = recall_score(y_valid, y_pred,  average='weighted')
        print(results)
        # Append results
        results.loc[i] = [clf.__class__.__name__, random_search.best_params_, acc, f1, precision, recall]

    return results
