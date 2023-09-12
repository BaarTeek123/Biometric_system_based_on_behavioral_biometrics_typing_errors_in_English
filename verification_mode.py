import os
from os.path import isdir, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NearMiss
from keras.metrics import AUC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample, shuffle
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier
from sklearn.base import BaseEstimator
from keras.models import Model as TFModel


from create_model import create_dataset
from cv import calculate_metrics
from decorators import log_info
from draw_results import plot_curves_and_matrix
# draw_system_t_roc_curve, draw_system_roc_curve
from logger import logger


def predict_input_fn(dataset):
    return dataset.map(lambda features, labels: features)

@log_info
def run_cv_tf(model, X, y, X_valid, y_valid, n_splits=5, name='',
              plot_path='results_identification/', title=None, use_weigted_dataset=False):
    kf = KFold(n_splits=n_splits)
    results = []
    # Initialization
    all_far = []
    all_frr = []
    all_tpr = []
    all_eer = []
    cumulative_cm = np.zeros((2, 2))
    threshold = None
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

        predictions = model.predict(X_valid)

        predictions = np.hstack((1 - predictions, predictions))
        y_score = predictions[:, 1]
        # Compute ROC curve metrics
        far, tpr, threshold = roc_curve(y_valid, y_score)
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
        results.append(calculate_metrics(y_valid, np.argmax(predictions, axis=1), 0.5, name=model.__class__.__name__))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i, (far, tpr) in enumerate(zip(all_far, all_tpr)):
        ax[0].plot(far, tpr, lw=1, alpha=0.3, label=f'Fold {i + 1} (EER = {all_eer[i]:.2f})')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curves')
    ax[0].legend(loc='lower right')

    for i, (far, frr) in enumerate(zip(all_far, all_frr)):
        ax[1].plot(frr, threshold, color='green', linewidth=2)
        ax[1].plot(frr, threshold, color='gray', linewidth=2)
        ax[1].set_title('T-ROC Curve')
        ax[1].set_xlabel('FRR (False Rejection Rate)')
        ax[1].set_ylabel('FAR (False Acceptance Rate)')

    sns.heatmap(cumulative_cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Other user', 'User'],
                yticklabels=['Other user', 'User'], ax=ax[2])
    ax[2].set_title('Confusion Matrix')
    ax[2].set_xlabel('Predicted')
    ax[2].set_ylabel('Actual')
    plt.tight_layout()
    if title is not None:
        if not isdir(plot_path):
            os.mkdir(plot_path)
        plt.savefig(join(plot_path, title + '.png'))
    plt.show()
    return pd.DataFrame(results)
    # return pd.DataFrame(results), all_far, all_frr, all_tpr, all_eer, cumulative_cm


def run_cv_sklearn(model, X, y, X_valid, y_valid, n_splits=5, name='',
                   plot_path='results_identification/', title=None, use_weigted_dataset=False):
    kf = KFold(n_splits=n_splits)
    results = []
    all_far = []
    all_frr = []
    all_tpr = []
    all_eer = []
    cumulative_cm = np.zeros((2, 2))
    thresholds = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if use_weigted_dataset:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_weight_dict = dict(zip(unique_classes, len(y_train) / (len(unique_classes) * class_counts)))
            sample_weights = np.array([class_weight_dict[class_val] for class_val in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        predictions = model.predict_proba(X_valid)[:, 1]

        # Compute ROC curve metrics
        far, tpr, thresh= roc_curve(y_valid, predictions)
        frr = 1 - tpr
        all_frr.append(frr)
        all_far.append(far)
        all_tpr.append(tpr)
        thresholds.append(thresh)

        # Compute EER
        eer_index = np.nanargmin(np.absolute(far - frr))
        eer = (far[eer_index] + frr[eer_index]) / 2
        all_eer.append(eer)
        threshold = 0.5
        cm = confusion_matrix(y_valid, (predictions >= threshold).astype(int))
        cumulative_cm += cm
        results.append(calculate_metrics(y_valid, (predictions >= threshold).astype(int), 0.5, model.__class__.__name__))
    plot_curves_and_matrix(all_far, all_tpr, all_eer, all_frr, thresholds, cumulative_cm, title=title, plot_path=plot_path)
    return pd.DataFrame(results)


"""
def balance_dataset(X, y, user, ratio = 1.0, random_state=42):
    user_mask = (y == user)
    non_user_mask = (y != user)

    X_balanced = np.concatenate([X[user_mask],
                                 resample(X[non_user_mask],
                                          n_samples=int(user_mask.sum()//ratio),
                                          random_state=random_state)])
    y_balanced = np.concatenate([np.ones(user_mask.sum()),
                                 np.zeros(X_balanced.shape[0] - user_mask.sum())])

    return shuffle(X_balanced, y_balanced, random_state=random_state)

@log_info
def train_and_evaluate_for_user(model, params, X_train, y_train, X_test, y_test, user, ratio=0.5, name:str= "", plot_path = 'results_verification/'):
    X_balanced_train, y_balanced_train = balance_dataset(X_train, y_train, user, ratio)
    X_balanced_test, y_balanced_test = balance_dataset(X_test, y_test, user)

    return run_cv_tf(model, X_balanced_train, y_balanced_train, X_balanced_test, y_balanced_test,
                     plot_path=plot_path, is_multiclass=False, name=name, user_name=f'user {user}')

"""


def oversample_dataset(X, y, user_label, ratio=0.3):
    """
    Balance the dataset using BorderlineSMOTE.

    Parameters:
    - X: Features
    - y: Labels
    - user: The user for which the dataset needs to be balanced
    - ratio: The desired ratio of minority to majority class after balancing

    Returns:
    - X_resampled: Resampled features
    - y_resampled: Resampled labels
    """
    user_mask = (y == user_label)
    non_user_mask = (y != user_label)
    print(f"User label {user_label}")

    num_of_non_user_samples = int(user_mask.sum()//ratio)
    if  num_of_non_user_samples > (y != user_label).sum()//ratio:
        num_of_non_user_samples =  int(0.8 * (y != user_label).sum())

    # assign 1 to user, 0 to others
    y = np.where(y == user_label, 1, 0)
    sampling_strategy = {0: num_of_non_user_samples}

    nearmiss = NearMiss(sampling_strategy=sampling_strategy)

    X, y = nearmiss.fit_resample(X, y)

    # X_balanced = np.concatenate([X[user_mask],
    #                              resample(X[non_user_mask],
    #                                       n_samples=int(user_mask.sum()//ratio),
    #                                       random_state=random_state)])
    # y_balanced = np.concatenate([np.ones(user_mask.sum()),
    #                              np.zeros(X_balanced.shape[0] - user_mask.sum())])

    # if not oversample:
    #     return X_balanced, y_balanced

    # Set the sampling strategy
    # sampling_strategy = {1: int((len(y) - sum(user_mask)) * ratio / (1 - ratio))}
    sampling_strategy = {1: num_of_non_user_samples}
    smote = BorderlineSMOTE(sampling_strategy=sampling_strategy)
    X, y = smote.fit_resample(X, y)

    return X, y

def balance_dataset(X, y, user_label, imbalance_ratio=0.3):
    y_labels = (y == user_label).sum()
    y = np.where(y == user_label, 1, 0)
    logger.info(f'Created {(y == 1).sum()} from {y_labels}')
    X_zero = X[y == 0]
    y_zero = y[y == 0]
    test_size = 1-((y == 1).sum() / (y == 0).sum())/imbalance_ratio
    if test_size < 0.01:
        test_size=0.01
    X_train_zero, X_test_zero, y_train_zero, y_test_zero = train_test_split(X_zero, y_zero,
                                                                            test_size=test_size)


    # Merge the training data
    return np.vstack((X_train_zero, X[y == 1])), np.hstack((y_train_zero, y[y == 1]))




def train_and_evaluate_for_user(model, params, X_train, y_train, X_test, y_test, user, oversample,
                                plot_path='results_verification/', title=None):
    if oversample:
        X_balanced_train, y_balanced_train = oversample_dataset(X_train, y_train, ratio=0.5, user_label=user,)
    else:
        X_balanced_train, y_balanced_train = balance_dataset(X_train, y_train, user_label=user)
    X_balanced_test, y_balanced_test = balance_dataset(X_test, y_test, user_label=user)
    common_args = {
        'X': X_balanced_train,
        # 'params': params,
        'y': y_balanced_train,
        'X_valid': X_balanced_test,
        'y_valid': y_balanced_test,
        'plot_path': plot_path,
        'use_weigted_dataset': False,
        'title': title
    }
    if isinstance(model, TFModel):
        return run_cv_tf(model, **common_args)
    elif isinstance(model, BaseEstimator):
        return run_cv_sklearn(model, **common_args)


if __name__ == '__main__':
    NUMBER_OF_FEATURES = 5
    N_GRAM_SIZE = 2
    OVERSAMPLE = False
    X, y, X_test, y_test = create_dataset(if_separate_words=True, test_ratio=0.5, verbose_mode=False,
                                                scaler=Normalizer(), n_gram_size=N_GRAM_SIZE, number_of_features=NUMBER_OF_FEATURES)
    model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1",
                                                 task=tfdf.keras.Task.CLASSIFICATION,
                                                 # tuner=tuner,
                                                 l2_regularization=0.01)

    final_df_list = []

    CLASSIFIERS =  [
        RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R', random_state=0),
        BalancedRandomForestClassifier(n_estimators=100, random_state=0),
        BalancedBaggingClassifier(estimator=DecisionTreeClassifier(),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=0)
    ]
    for model in CLASSIFIERS:
        final_df_list = []
        for user in np.unique(y):
            logger.info(f'Starting train & evaluation process with {user}')
            print()
            final_df_list.append(train_and_evaluate_for_user(model, None, X, y, X_test, y_test, user=user,
                                                             plot_path=f'results_verification/{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}',oversample=OVERSAMPLE,
                                                             title=f'{model.__class__.__name__}_undersample_{user}'))

        df = pd.concat(final_df_list, ignore_index=True)
        df['Number of features'] = NUMBER_OF_FEATURES
        df['N-gram sizes'] = N_GRAM_SIZE
        df['Classifier'] = model.__class__.__name__
        df['Oversample'] = OVERSAMPLE
        filename=f'results_verification/{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}.csv'
        df.to_csv(filename, mode='a+', header=not os.path.exists(filename) or pd.read_csv(filename).empty)