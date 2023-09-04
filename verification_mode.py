import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import Normalizer
import tensorflow_decision_forests as tfdf
from sklearn.metrics import confusion_matrix

from create_model import NUMBER_OF_FEATURES, N_GRAM_SIZE, create_dataset
from cv import calculate_metrics
from decorators import log_info
from draw_results import calculate_far_frr_eer, draw_system_t_roc_curve, plot_cmc, draw_system_roc_curve, \
    draw_classes_roc_curve, plot_confusion_metrics
# draw_system_t_roc_curve, draw_system_roc_curve
from logger import logger

def predict_input_fn(dataset):
    return dataset.map(lambda features, labels: features)

@log_info
def run_cv_tf(model, X, y, X_valid, y_valid, n_splits=5, name='',
           plot_path='results_identification/', user_name = None, is_multiclass=True):

    kf = KFold(n_splits=n_splits)
    results = []

    for train_index, test_index in kf.split(X):
        # Create train and test datasets using the indices from KFold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(1000)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1000)
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(1000)
        model.fit(train_dataset)
        predictions = model.predict(X_valid, verbose=2)
        predictions = np.hstack((1 - predictions, predictions))
        cm = confusion_matrix(y_valid, np.argmax(predictions, axis=1))
        print(cm)
        if user_name is None:
                user_name = 'User'
        plot_confusion_metrics(y_valid, predictions, ['No user', user_name])
        far, frr, eer, threshold = calculate_far_frr_eer(y_valid, predictions)

        draw_system_roc_curve(far, frr, eer, plot_title="Receiver operating characteristic", file_path=f'{plot_path}/{name}_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}_far_frr.png')
        draw_system_t_roc_curve(far, frr, eer, plot_title="T-Receiver operating characteristic",
                                file_path=f'{plot_path}/{name}_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}_t_roc.png')
        # draw_system_roc_curve(far, frr, eer, plot_title="Receiver operating characteristic ",
        #                       file_path=f'{plot_path}/{name}_{NUMBER_OF_FEATURES}_{N_GRAM_SIZE}_roc.png')

    return pd.DataFrame(results)

"""
        # y_pred = model.predict_proba(valid_dataset)
        threshold = 0.02
        probabilities = []
        for pred_dict in predictions:
            prob = pred_dict['probabilities'][1]  # Probability for class 1
            probabilities.append(prob)
        for k in range(2, 51):
            y_pred = (y_pred >= threshold) * y_pred
            non_zero_mask = ~(np.all(y_pred == 0, axis=1))
            rejected = y_pred[~non_zero_mask].shape[0] / y_pred.shape[0]
            threshold = 0.02 * k

            if is_multiclass:
                cmc = calculate_cmc(y_valid, np.array(y_pred))
                print(cmc)
                plot_cmc(cmc, file_path=f'{plot_path}/{name}_{k}_cmc.png',
                             plot_title="Cumulative Match Characteristic (CMC) Curve")

            y_pred_labels = np.array([np.argmax(p) if np.max(p) > threshold else -1 for p in y_pred])

            if is_multiclass:
                results.append(calculate_metrics(y_valid, y_pred_labels, threshold, name, cmc=cmc, rejected=rejected))
            else:
                results.append(calculate_metrics(y_valid, y_pred_labels, threshold, name, rejected=rejected))
"""




def balance_dataset(X, y, user, random_state=42):
    user_mask = (y == user)
    non_user_mask = (y != user)

    X_balanced = np.concatenate([X[user_mask],
                                 resample(X[non_user_mask],
                                          n_samples=user_mask.sum(),
                                          random_state=random_state)])
    y_balanced = np.concatenate([np.ones(user_mask.sum()),
                                 np.zeros(user_mask.sum())])
    return shuffle(X_balanced, y_balanced, random_state=random_state)


@log_info
def train_and_evaluate_for_user(model, params, X_train, y_train, X_test, y_test, user):
    X_balanced_train, y_balanced_train = balance_dataset(X_train, y_train, user)
    X_balanced_test, y_balanced_test = balance_dataset(X_test, y_test, user)


    return run_cv_tf(model, X_balanced_train, y_balanced_train, X_balanced_test, y_balanced_test, plot_path='results_verification/', is_multiclass=False, name='GBDT', user_name=f'user {user}')



if __name__ == '__main__':
    X, y, X_test, y_test, cols = create_dataset(if_separate_words=True, test_ratio=0.5, verbose_mode=True,
                                                scaler=Normalizer())
    model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1",
                                                 task=tfdf.keras.Task.CLASSIFICATION,
                                                 # tuner=tuner,
                                                 l2_regularization=0.01)
    final_df_list = []
    for user in np.unique(y):
        final_df_list.append(train_and_evaluate_for_user(model, None, X, y, X_test, y_test, user))
# import time
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd
# import classifiers
# from classifiers import build_tuned_nn, build_tuned_rfc, param_grid
# from sklearn.neural_network import MLPClassifier
# from create_model import create_dataset
# from sklearn.svm import SVC
# from sklearn.utils import shuffle
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
#     classification_report
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.utils import resample
# import tensorflow_decision_forests as tfdf
# import tensorflow as tf
#
# from draw_results import calculate_cmc
#
#
#
#
#
# if __name__ == "__main__":
#
#     CLASSIFIERS = [
#         # (build_tuned_rfc,
#         #  {'clf': RandomForestClassifier(), 'x_train': X_valid, 'y_train': y_valid, 'param_grid':
#         (RandomForestClassifier(), param_grid['Random Forest'],
#          'Random Forest'),
#
#         (KNeighborsClassifier(),
#          param_grid['K-Nearest Neighbors'],
#          'K-Nearest Neighbors'),
#
#         (SVC(probability=True),
#          param_grid['SVC'],
#          'SVC'),
#
#         (GradientBoostingClassifier(),
#          param_grid['Gradient Boosting'],
#          'Gradient Boosting'),
#
#         (MLPClassifier(),
#          param_grid['MLP Classifier'],
#          'MLP Classifier')
# ]
#
#
#     X_train, y_train, X_test, y_test = create_dataset(test_ratio=0.5, if_separate_words=True, n_gram_size=2, number_of_features=5, amount_of_n_grams_pers_user=3100, )
#     history_dict = {}
#     df_list = []
#     for user in np.unique(y_train):
#
#         # Create balanced datasets for user and non-user
#         user_mask_train = (y_train == user)
#         non_user_mask_train = (y_train != user)
#
#         user_mask_test = (y_test == user)
#         non_user_mask_test = (y_test != user)
#
#         # Concatenate user and non-user samples (equal numbers)
#         X_balanced_train = np.concatenate([X_train[user_mask_train],
#                                            resample(X_train[non_user_mask_train],
#                                                     n_samples=user_mask_train.sum(),
#                                                     random_state=42)])
#
#         y_balanced_train = np.concatenate([np.ones(user_mask_train.sum()),
#                                            np.zeros(user_mask_train.sum())])  # 1 for user, 0 for non-user
#
#         X_balanced_test = np.concatenate([X_test[user_mask_test],
#                                           resample(X_test[non_user_mask_test],
#                                                    n_samples=user_mask_test.sum(),
#                                                    random_state=42)])
#
#         y_balanced_test = np.concatenate([np.ones(user_mask_test.sum()),
#                                           np.zeros(user_mask_test.sum())])  # 1 for user, 0 for non-user
#
#         # Shuffling the balanced train and test sets
#         X_balanced_train, y_balanced_train = shuffle(X_balanced_train, y_balanced_train, random_state=42)
#         X_balanced_test, y_balanced_test = shuffle(X_balanced_test, y_balanced_test, random_state=42)
#
#         # Create and compile the model
#         # model = KerasNNClf(X_train.shape[1], len(user_names.keys()), True).create_neural_network()
#         # model = classifiers.create_neural_network(X_train.shape[1], len(user_names.keys()), True)
#
#         model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
#
#         train_ds = tf.data.Dataset.from_tensor_slices((X_balanced_train, y_balanced_train)).batch(1000)
#         test_ds = tf.data.Dataset.from_tensor_slices((X_balanced_test, y_balanced_test)).batch(1000)
#         history_dict = {}
#
#         # class_weights = (np.bincount(y_train.astype(int)).max() / np.bincount(y_train.astype(int))).tolist()
#         tuner = tfdf.tuner.RandomSearch(num_trials=15)
#         tuner.choice("min_examples", [2, 5, 7, 10])
#         tuner.choice("categorical_algorithm", ["CART", "RANDOM"])
#         global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"])
#         global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])
#         tuner.choice("use_hessian_gain", [True, False])
#         tuner.choice("shrinkage", [0.02, 0.05, 0.10, 0.15])
#         tuner.choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])
#
#         model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1",
#                                                      task=tfdf.keras.Task.CLASSIFICATION, tuner=tuner,
#                                                      l2_regularization=0.01)
#
#         model.compile(metrics=["accuracy"])
#
#         # Raise the roof with some training!
#         model.fit(train_ds)
#         y_pred = model.predict(test_ds, use_multiprocessing=True, workers=5).ravel()
#
#
#         for i in range(10):
#             y_pred_class = [1. if prob >= (i+1)/10 else 0. for prob in y_pred]
#             print(f"Threshold: {(i+1)/10}", confusion_matrix(y_balanced_test, y_pred_class))
#             report_df = pd.concat([pd.DataFrame(classification_report(y_balanced_test, y_pred_class, output_dict=True)).transpose()], keys=[f'{user} - Threshold: {(i+1)/10}'], names=['User-threshold'])
#             df_list.append(report_df)
#             print(report_df)
#     final_df = pd.concat(df_list)
#     final_df['clf']='Neural network'
#     final_df.to_csv('verification_test.csv')
#
#     # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=365)
#     # for user in np.unique(y_train):
#     #     # Create balanced datasets for user and non-user
#     #     user_mask_train = (y_train == user)
#     #     non_user_mask_train = (y_train != user)
#     #
#     #     user_mask_test = (y_test == user)
#     #     non_user_mask_test = (y_test != user)
#     #
#     #     # Concatenate user and non-user samples (equal numbers)
#     #     X_balanced_train = np.concatenate([X_train[user_mask_train],
#     #                                        resample(X_train[non_user_mask_train],
#     #                                                 n_samples=user_mask_train.sum(),
#     #                                                 random_state=42)])
#     #
#     #     y_balanced_train = np.concatenate([np.ones(user_mask_train.sum()),
#     #                                        np.zeros(user_mask_train.sum())])  # 1 for user, 0 for non-user
#     #
#     #     X_balanced_test = np.concatenate([X_test[user_mask_test],
#     #                                       resample(X_test[non_user_mask_test],
#     #                                                n_samples=user_mask_test.sum(),
#     #                                                random_state=42)])
#     #
#     #     y_balanced_test = np.concatenate([np.ones(user_mask_test.sum()),
#     #                                       np.zeros(user_mask_test.sum())])  # 1 for user, 0 for non-user
#     #
#     #     # Shuffling the balanced train and test sets
#     #     X_balanced_train, y_balanced_train = shuffle(X_balanced_train, y_balanced_train, random_state=42)
#     #     X_balanced_test, y_balanced_test = shuffle(X_balanced_test, y_balanced_test, random_state=42)
#     #
#     #     # Create and compile the model
#     #     model = KerasNNClf(X_train.shape[1], len(user_names.keys()), True).create_neural_network()
#     #
#     #     # Train the model with fewer epochs and a larger batch size
#     #     history = model.fit(X_balanced_train, y_balanced_train, epochs=100, batch_size=64, callbacks=[model.earlystopping, model.logger])
#     #
#     #     # Evaluate the model
#     #     loss, accuracy = model.evaluate(X_balanced_test, y_balanced_test)
#     #
#     #     print('Test loss:', loss)
#     #     print('Test accuracy:', accuracy)
#     #
#     #     # Generate and print the confusion matrix
#     #     y_pred = model.predict(X_balanced_test).ravel()
#     #
#     #     y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred]
#     #     print(confusion_matrix(y_balanced_test, y_pred_class))
#
#     # unique_users_train = np.unique(y_train)
#     # unique_users_test = np.unique(y_test)
#     # y_train = label_binarize(y_train, classes=unique_users_train.tolist())
#     # y_test = label_binarize(y_test, classes=unique_users_test.tolist())
#     #
#     # n_classes_train = y_train.shape[1]
#     # n_classes_test = y_test.shape[1]
#     #
#     # # Learn to predict each class against the other using RandomForest
#     # classifier = OneVsRestClassifier(MLPClassifier())
#     # classifier.fit(X_train, y_train)
#     #
#     # # Predict the classes
#     # y_score = classifier.predict(X_test)
#     #
#     # print(y_score)
#     #
#     # # Precision, Recall, F1-score
#     # precision = precision_score(y_test, y_score, average='micro')
#     # recall = recall_score(y_test, y_score, average='micro')
#     # f1 = f1_score(y_test, y_score, average='micro')
#     #
#     # print("Precision: ", precision)
#     # print("Recall: ", recall)
#     # print("F1-score: ", f1)
#     #
#     # # ROC AUC
#     # # For ROC AUC, we need probability estimates of the positive class
#     # # The classifier should have predict_proba method
#     #
#     # y_score_proba = classifier.predict_proba(X_test)
#     # roc_auc = roc_auc_score(y_test, y_score_proba, multi_class='ovr', average='micro')
#     #
#     # print("ROC AUC: ", roc_auc)
