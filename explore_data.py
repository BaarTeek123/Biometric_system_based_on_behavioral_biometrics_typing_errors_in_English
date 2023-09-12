import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from keras.metrics import AUC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from os.path import isdir, join

from create_model import create_dataset
from logger import logger
# from verification_mode import balance_dataset

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


from imblearn.over_sampling import BorderlineSMOTE


def balance_dataset(X, y, user, ratio=0.2):
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

    # Assuming 'user' is the minority class and '1-user' is the majority class
    # Calculate the number of samples for the minority class after resampling
    n_minority = sum(y == user)
    n_majority = len(y) - n_minority
    desired_n_minority = int(n_majority * ratio / (1 - ratio))

    # Set the sampling strategy
    sampling_strategy = {user: desired_n_minority}

    smote = BorderlineSMOTE(sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


def train_and_evaluate_for_user(model, params, X_train, y_train, X_test, y_test, user,
                                plot_path='results_verification/', title=None):
    X_balanced_train, y_balanced_train = balance_dataset(X_train, y_train, user, ratio=0.2)
    X_balanced_test, y_balanced_test = balance_dataset(X_test, y_test, user)

    return run_cv_tf(model, X_balanced_train, y_balanced_train, X_balanced_test, y_balanced_test, plot_path=plot_path,
                     use_weigted_dataset=False, title=title)


if __name__ == '__main__':
    NUMBER_OF_FEATURES = 5
    N_GRAM_SIZE = 2
    X, y, X_test, y_test, cols = create_dataset(if_separate_words=True, test_ratio=0.5, verbose_mode=False,
                                                n_gram_size=N_GRAM_SIZE, number_of_features=NUMBER_OF_FEATURES,
                                                amount_of_n_grams_pers_user=70000, scaler=Normalizer())

    # %%
    model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1",
                                                 task=tfdf.keras.Task.CLASSIFICATION,
                                                 # tuner=tuner,
                                                 l2_regularization=0.01)
    final_df_list = []
    all_far, all_frr, all_tpr, all_eer = [], [], [], []
    cumulative_cm = np.zeros((2, 2))
    for user in np.unique(y):
        res = train_and_evaluate_for_user(model, None, X, y, X_test, y_test, user)
