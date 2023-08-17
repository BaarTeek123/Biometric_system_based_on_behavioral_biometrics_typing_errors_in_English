from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from tensorflow import expand_dims
import numpy as np
from classifiers import create_neural_network, earlystopping, logger
from sklearn.preprocessing import LabelBinarizer
from keras.models import clone_model
from n_grams_creator import user_names
from random import sample
import matplotlib.pyplot as plt

def run_cv(clf, params, X, y, X_valid, y_valid, n_splits=5, n_repeats=2, threshold=0.3, predef_model=False,
           is_categorical=False):
    # Initialize RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    results = []

    for i, (train_index, test_index) in enumerate(rskf.split(X, y), 1):
        # Select training set for this fold
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if not predef_model:
            # Create a fresh clone of the classifier for this repeat
            clf_clone = clone(clf)
            # Tune hyperparameters on the current fold
            random_search = RandomizedSearchCV(clf_clone, params, n_jobs=-1, cv=3)
            random_search.fit(X_train, y_train)
            best_clf = random_search.best_estimator_

        else:
            # Use the predefined model
            best_clf = clone(clf)
            best_clf.compile(optimizer='adam', loss='binary_crossentropy')
            best_clf.fit(X_train, y_train, epochs=10, verbose=0)
        best_clf.fit(X_test, y_test)
        # Calculate probabilities on the validation set
        probabilities = best_clf.predict_proba(X_valid)
        threshold = 0.02
        for k in range(2, 51):
            y_pred = [np.argmax(p) if np.max(p) > threshold else -1 for p in probabilities]

            acc = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
            precision = precision_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
            recall = recall_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
            # Append to results
            results.append(
                {'clf': 'Neural Network', 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
                 'iteration': k, 'threshold': threshold})
            threshold = 0.02 * k
        print(pd.DataFrame(results))
    return pd.DataFrame(results)


# def run_cv_neural_network(X, y, n_splits=5, n_repeats=2):
#
#     lb = LabelBinarizer()
#     y_binarized = lb.fit_transform(y)
#
#     # Create a DataFrame to store the results
#     results = []
#
#     # Create a RepeatedStratifiedKFold object
#     rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
#     prototype_model = create_neural_network(input_dim=X.shape[1], output_dim=len(user_names.keys()), binary=False)
#     # Loop through the splits
#
#     for k, (train_index, test_index) in enumerate(rskf.split(X, y), 1):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y_binarized[train_index], y_binarized[test_index]
#         clf = clone_model(prototype_model)
#         # clf.compile(optimizer=prototype_model.optimizer, loss=prototype_model.loss,
#         #               metrics=prototype_model.metrics)
#         clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#         clf.fit(expand_dims(X_train, axis=-1), y_train, epochs=50, batch_size=128, callbacks=[earlystopping, logger])
#
#         # Make predictions
#         y_pred = clf.predict(X_test)
#         threshold = 0.02
#         for i in range(2, 51):
#             y_pred = (y_pred >= threshold) * y_pred
#             # Convert predictions to labels
#             y_pred_labels = lb.inverse_transform(y_pred)
#             y_test_labels = lb.inverse_transform(y_test)
#
#             acc = accuracy_score(y_test_labels, y_pred_labels)
#             f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
#             precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
#             recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
#
#             # Append to results
#             results.append(
#                 {'clf': 'Neural Network', 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
#                  'iteration': k, 'threshold': threshold})
#             threshold = 0.02 * i
#
#         # Compute metrics
#
#     return pd.DataFrame(results)

def calculate_cmc(y_test, probs, threshold=0.0):
    sorted_indices = np.argsort(-probs, axis=1)

    # Initialize ranks array with a value out of the normal rank range (e.g., 9 for an 8 class problem)
    ranks = np.full(y_test.shape, 9)

    for i in range(8):  # adjust this based on the number of classes or columns in your probs array
        # Find indices where the true label matches the sorted index and the probability is above the threshold
        mask = (sorted_indices[:, i] == y_test) & (probs[np.arange(probs.shape[0]), sorted_indices[:, i]] >= threshold)
        ranks[mask] = i

    # Compute CMC curve
    cmc_counts = np.bincount(ranks, minlength=9)[:8]
    return np.cumsum(cmc_counts) / len(y_test)


def plot_cmc(cmc_curve, file_path=None):
    # Plotting the CMC curve
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    plt.plot(cmc_curve, marker='o', color='royalblue', linestyle='-', markersize=4)
    plt.title("Cumulative Match Characteristic (CMC) Curve", fontsize=16, fontweight='bold')
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Recognition Rate", fontsize=14)
    plt.xticks(np.arange(0, 8, 1))  # Adjust according to your dataset's number of classes
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
    # Show plot
    plt.show()


def run_cv_neural_network(X, y, X_test=None, y_test=None, n_splits: int=5, n_repeats: int=2):

    lb = LabelBinarizer()
    y_binarized = lb.fit_transform(y)

    # Create a DataFrame to store the results
    results = []

    # If X_test and y_test are not provided, perform cross-validation
    if X_test is None or y_test is None:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        splits = rskf.split(X, y)
    else:
        splits = [
            [
                sample(range(len(X)), int(0.9 * len(X))),
                sample(range(len(X_test)), int(0.9 * len(X_test))),
            ]
            for _ in range(int(n_splits * n_repeats))
        ]

    prototype_model = create_neural_network(input_dim=X.shape[1], output_dim=len(user_names.keys()), binary=False)

    # Loop through the splits
    for k, (train_index, test_index) in enumerate(splits, 1):
        print(k)
        if X_test is not None and y_test is not None:
            X_train, X_val = X[train_index], X_test
            y_train, y_val = y_binarized[train_index], lb.transform(y_test)
        else:
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y_binarized[train_index], y_binarized[test_index]
        clf = clone_model(prototype_model)
        clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        clf.fit(expand_dims(X_train, axis=-1), y_train, epochs=50, batch_size=128, callbacks=[earlystopping, logger])

        # Make predictions
        y_pred = clf.predict(X_val)
        cmc = calculate_cmc(y_test, y_pred)
        print(cmc)
        plot_cmc(cmc,  'plots/random/cmc.png')
        threshold = 0.02
        for i in range(2, 51):
            y_pred = (y_pred >= threshold) * y_pred
            non_zero_mask = ~(np.all(y_pred == 0, axis=1))

            rejected = y_pred[~non_zero_mask].shape[0] /  y_pred.shape[0]
            cmc = calculate_cmc(y_test, y_pred)
            print(cmc)
            if i%10 == 0:
                plot_cmc(cmc, f'plots/random/{i}_cmc.png')
            # Convert predictions to labels
            y_pred_labels = lb.inverse_transform(y_pred)
            y_val_labels = lb.inverse_transform(y_val)
            # set -1 for those thresh > prediction
            y_pred_labels[~non_zero_mask] = -1


            acc = accuracy_score(y_val_labels, y_pred_labels)
            f1 = f1_score(y_val_labels, y_pred_labels, average='weighted')
            precision = precision_score(y_val_labels, y_pred_labels, average='weighted')
            recall = recall_score(y_val_labels, y_pred_labels, average='weighted')

            # Append to results
            results.append(
                {'clf': 'Neural Network', 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
                 'iteration': k, 'threshold': threshold, 'rejected': rejected})
            threshold = 0.02 * i
        print(pd.DataFrame(results))

    return pd.DataFrame(results)
