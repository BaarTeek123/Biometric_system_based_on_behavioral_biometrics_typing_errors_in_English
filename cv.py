from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from tensorflow import expand_dims
import numpy as np
from classifiers import create_neural_network, earlystopping, logger
from sklearn.preprocessing import LabelBinarizer
from create_model import N_GRAM_SIZE, NUMBER_OF_FEATURES
from draw_results import plot_result_nn, calculate_far_frr_eer, draw_classes_roc_curve, calculate_cmc, plot_cmc, \
    draw_far_frr, draw_system_t_roc_curve, draw_system_roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import pandas as pd
from random import sample
from keras.models import clone_model
from n_grams_creator import user_names
from random import sample
import matplotlib.pyplot as plt
def calculate_metrics(y_valid, y_pred, threshold, name="" ,**kwargs):

    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
    precision = precision_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
    # Append to results
    return {'clf': name, 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
         'threshold': threshold, **kwargs}

# def run_cv(clf, params, X, y, X_valid, y_valid, n_splits=5, n_repeats=2 ,predef_model=False, name='',
#            plot_path='results_identification/', is_multiclass=True):
#     # Initialize RepeatedStratifiedKFold
#     rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
#     results = []
#
#     for i, (train_index, test_index) in enumerate(rskf.split(X, y), 1):
#         # Select training set for this fold
#         X_train, y_train = X[train_index], y[train_index]
#         X_test, y_test = X[test_index], y[test_index]
#         if not predef_model:
#             # Create a fresh clone of the classifier for this repeat
#             clf_clone = clone(clf)
#             # Tune hyperparameters on the current fold
#             random_search = RandomizedSearchCV(clf_clone, params, n_jobs=-1, cv=3)
#             random_search.fit(X_train, y_train)
#             best_clf = random_search.best_estimator_
#
#         else:
#             # Use the predefined model
#             best_clf = clone(clf)
#             best_clf.fit(X_train, y_train)
#         best_clf.fit(X_test, y_test)
#         # Calculate probabilities on the validation set
#         y_pred = best_clf.predict_proba(X_valid)
#         threshold = 0.02
#
#         for k in range(2, 51):
#             y_pred = (y_pred >= threshold) * y_pred
#             non_zero_mask = ~(np.all(y_pred == 0, axis=1))
#             rejected = y_pred[~non_zero_mask].shape[0] / y_pred.shape[0]
#             threshold = 0.02 * k
#             if is_multiclass:
#                 cmc = calculate_cmc(y_valid, np.array(y_pred))
#                 print(cmc)
#                 if i % 10 == 0:
#                     plot_cmc(cmc, file_path=f'{plot_path}/{name}_{i}_{k}_cmc.png',
#                              plot_title="Cumulative Match Characteristic (CMC) Curve")
#             y_pred_labels = np.array([np.argmax(p) if np.max(p) > threshold else -1 for p in y_pred])
#             if is_multiclass:
#                 results.append(calculate_metrics(y_valid, y_pred_labels, threshold, name, cmc=cmc, rejected=rejected))
#             else:
#                 results.append(calculate_metrics(y_valid, y_pred_labels, threshold, name,  rejected=rejected))
#
#
#         print(pd.DataFrame(results))
#     return pd.DataFrame(results)
#
#
#


from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
import numpy as np
import pandas as pd


def run_cv(clf, params, X, y, X_valid, y_valid, n_splits=5, n_repeats=2, predef_model=False, name='',
           plot_path='results_identification/', is_multiclass=True):
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
            best_clf.fit(X_train, y_train)

        best_clf.fit(X_test, y_test)

        # Check if the classifier supports predict_proba
        if hasattr(best_clf, "predict_proba"):
            y_pred = best_clf.predict_proba(X_valid)
        else:
            y_pred = best_clf.predict(X_valid)
            y_pred = np.eye(y_pred.max() + 1)[y_pred]  # Convert labels to one-hot encoded format

        threshold = 0.02

        for k in range(2, 51):
            y_pred = (y_pred >= threshold) * y_pred
            non_zero_mask = ~(np.all(y_pred == 0, axis=1))
            rejected = y_pred[~non_zero_mask].shape[0] / y_pred.shape[0]
            threshold = 0.02 * k

            if is_multiclass:
                cmc = calculate_cmc(y_valid, np.array(y_pred))
                print(cmc)
                if i % 10 == 0:
                    plot_cmc(cmc, file_path=f'{plot_path}/{name}_{i}_{k}_cmc.png',
                             plot_title="Cumulative Match Characteristic (CMC) Curve")

            y_pred_labels = np.array([np.argmax(p) if np.max(p) > threshold else -1 for p in y_pred])

            if is_multiclass:
                results.append(calculate_metrics(y_valid, y_pred_labels, threshold, name, cmc=cmc, rejected=rejected))
            else:
                results.append(calculate_metrics(y_valid, y_pred_labels, threshold, name, rejected=rejected))

        print(pd.DataFrame(results))
    return pd.DataFrame(results)


def run_cv_neural_network(X, y, X_test=None, y_test=None, n_splits: int=5, n_repeats: int=2, plot_path: str = 'results_identification/random/', epochs=50,
                          is_multiclass=True):

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
    if is_multiclass:
        prototype_model = create_neural_network(input_dim=X.shape[1], output_dim=len(user_names.keys()), binary=False)
    else:
        prototype_model = create_neural_network(input_dim=X.shape[1], output_dim=len(user_names.keys()), binary=True)


    # Loop through the splits
    for k, (train_index, test_index) in enumerate(splits, 1):
        if X_test is not None and y_test is not None:
            X_train, X_val = X[train_index], X_test
            y_train, y_val = y_binarized[train_index], lb.transform(y_test)
        else:
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y_binarized[train_index], y_binarized[test_index]
        clf = clone_model(prototype_model)
        clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = clf.fit(expand_dims(X_train, axis=-1), y_train, epochs=epochs, batch_size=128, callbacks=[earlystopping, logger])
        plot_result_nn(history, file_path=f'{plot_path}/neural_network_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}{k}_cmc.png')
        # Make predictions
        y_pred = clf.predict(X_val)
        if is_multiclass:
            cmc = calculate_cmc(y_test, y_pred)
            print(cmc)
            plot_cmc(cmc, file_path=f'{plot_path}/neural_network_no_thresh_cmc.png', plot_title="Cumulative Match Characteristic (CMC) Curve")
        far, frr, eer, threshold = calculate_far_frr_eer(y_test, y_pred)
        # draw_classes_roc_curve(y_test, y_pred, plot_title="Receiver operating characteristic",
        #                        classes=user_names,
        #                        file_path=f'{plot_path}/neural_network_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}{k}_class_roc.png')
        draw_far_frr(far, frr, eer, threshold, file_path=f'{plot_path}/neural_network_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}{k}_far_frr.png')
        draw_system_t_roc_curve(far, frr, eer, plot_title="Receiver operating characteristic",
                                file_path=f'{plot_path}/neural_network_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}{k}_t_roc.png')
        draw_system_roc_curve(far, frr, eer, plot_title="Receiver operating characteristic ",
                              file_path=f'{plot_path}/neural_network_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}{k}_roc.png')

        threshold = 0.02
        for i in range(2, 51):
            y_pred = (y_pred >= threshold) * y_pred
            non_zero_mask = ~(np.all(y_pred == 0, axis=1))
            rejected = y_pred[~non_zero_mask].shape[0] / y_pred.shape[0]
            if is_multiclass:
                cmc = calculate_cmc(y_test, y_pred)
                print(cmc)
                if i%10 == 0:
                    plot_cmc(cmc, file_path=f'{plot_path}/neural_network_{k}_{i}_cmc.png', plot_title="Cumulative Match Characteristic (CMC) Curve")
            # Convert predictions to labels
            y_pred_labels = lb.inverse_transform(y_pred)
            y_val_labels = lb.inverse_transform(y_val)
            # set -1 for those thresh > prediction
            y_pred_labels[~non_zero_mask] = -1
            if is_multiclass:
                results.append(calculate_metrics(y_val_labels, y_pred_labels, threshold, 'Neural network', cmc=cmc, rejected=rejected,
                                             far=far, frr=frr, eer=eer))
            else:
                results.append(calculate_metrics(y_val_labels, y_pred_labels, threshold, 'Neural network',  rejected=rejected))

            threshold = 0.02 * i
        print(pd.DataFrame(results))

    return pd.DataFrame(results)

