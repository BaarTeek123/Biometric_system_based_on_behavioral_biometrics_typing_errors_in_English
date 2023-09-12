from os.path import isdir, join
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.metrics import AUC
from keras.models import clone_model
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import expand_dims

from classifiers import create_neural_network, earlystopping, logger
from create_model import N_GRAM_SIZE, NUMBER_OF_FEATURES
from draw_results import plot_result_nn, calculate_far_frr_eer, calculate_cmc, plot_cmc, \
    draw_far_frr, draw_system_t_roc_curve, draw_system_roc_curve
from n_grams_creator import user_names


def calculate_metrics(y_valid, y_pred, threshold, name="" ,**kwargs):

    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
    precision = precision_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_valid, y_pred, average='weighted', labels=np.unique(y_pred))
    # Append to results
    return {'clf': name, 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
         'threshold': threshold, **kwargs}

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



### verification system - Tensorflow

def run_cv_tf(model, X, y, X_valid, y_valid, n_splits=5,
              plot_path='results_identification/', title=None, use_weigted_dataset=False):
    kf = KFold(n_splits=n_splits)
    results = []
    # Initialization
    all_far = []
    all_frr = []
    all_tpr = []
    all_eer = []
    cumulative_cm = np.zeros((2, 2))

    for train_index, test_index in kf.split(X):
        # Create train and test datasets using the indices from KFold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        if use_weigted_dataset:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_weight_dict = dict(zip(unique_classes, len(y_train) / (len(unique_classes) * class_counts)))
            sample_weights = np.array([class_weight_dict[class_val] for class_val in y_train])
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights)).batch(1000)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(1000)

        X_valid, no_use_x, y_valid, no_use_y = train_test_split(X_valid, y_valid, test_size=0.1)

        model.fit(train_dataset)
        model.compile(metrics=[AUC(name='auc')])
        predictions = model.predict(X_valid, verbose=2)
        predictions = np.hstack((1 - predictions, predictions))

        # predictions = model.predict(X_valid).reshape(-1, 1)
        predictions = model.predict(X_valid)

        predictions = np.hstack((1 - predictions, predictions))
        y_score = predictions[:, 1]
        # Compute ROC curve metrics
        far, tpr, _ = roc_curve(y_valid, y_score)
        frr = 1 - tpr
        all_frr.append(frr)
        all_far.append(far)
        all_tpr.append(tpr)
        # Compute EER
        eer_index = np.nanargmin(np.absolute(far - frr))
        eer = (far[eer_index] + frr[eer_index]) / 2
        all_eer.append(eer)

        cm = confusion_matrix(y_valid, np.argmax(predictions, axis=1))
        cumulative_cm += cm
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i, (far, tpr) in enumerate(zip(all_far, all_tpr)):
        ax[0].plot(far, tpr, lw=1, alpha=0.3, label=f'Fold {i + 1} (EER = {all_eer[i]:.2f})')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curves')
    ax[0].legend(loc='lower right')

    for i, (far, frr) in enumerate(zip(all_far, all_frr)):
        ax[1].plot(frr, far, color='green', linewidth=2)

        ax[1].set_title('T-ROC Curve')
        ax[1].set_xlabel('FRR (False Rejection Rate)')
        ax[1].set_ylabel('FAR (False Acceptance Rate)')

    sns.heatmap(cumulative_cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Other user', 'User'],
                yticklabels=['Other user', 'User'], ax=ax[2])
    ax[2].set_title('Confusion Matrix')
    ax[2].set_xlabel('Predicted')
    ax[2].set_ylabel('Actual')
    plt.tight_layout()
    if isdir(plot_path) and title is not None:
        plt.savefig(join(plot_path, title + '.png'))
    plt.show()

    return pd.DataFrame(results), all_far, all_frr, all_tpr, all_eer, cumulative_cm



