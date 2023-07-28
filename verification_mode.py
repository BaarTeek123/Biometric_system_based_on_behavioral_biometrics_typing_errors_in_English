import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from classifiers import build_tuned_nn, build_tuned_rfc, param_grid, create_neural_network
from sklearn.neural_network import MLPClassifier
from create_model import create_dataset, user_names
from sklearn.svm import SVC
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample


if __name__ == "__main__":

    CLASSIFIERS = [
        # (build_tuned_rfc,
        #  {'clf': RandomForestClassifier(), 'x_train': X_valid, 'y_train': y_valid, 'param_grid':
        (RandomForestClassifier(), param_grid['Random Forest'],
         'Random Forest'),

        (KNeighborsClassifier(),
         param_grid['K-Nearest Neighbors'],
         'K-Nearest Neighbors'),

        (SVC(probability=True),
         param_grid['SVC'],
         'SVC'),

        (GradientBoostingClassifier(),
         param_grid['Gradient Boosting'],
         'Gradient Boosting'),

        (MLPClassifier(),
         param_grid['MLP Classifier'],
         'MLP Classifier')
]


    X_train, y_train, X_test, y_test = create_dataset(test_ratio=0.5)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=365)
    for user in np.unique(y_train):
        # Create balanced datasets for user and non-user
        user_mask_train = (y_train == user)
        non_user_mask_train = (y_train != user)

        user_mask_test = (y_test == user)
        non_user_mask_test = (y_test != user)

        # Concatenate user and non-user samples (equal numbers)
        X_balanced_train = np.concatenate([X_train[user_mask_train],
                                           resample(X_train[non_user_mask_train],
                                                    n_samples=user_mask_train.sum(),
                                                    random_state=42)])

        y_balanced_train = np.concatenate([np.ones(user_mask_train.sum()),
                                           np.zeros(user_mask_train.sum())])  # 1 for user, 0 for non-user

        X_balanced_test = np.concatenate([X_test[user_mask_test],
                                          resample(X_test[non_user_mask_test],
                                                   n_samples=user_mask_test.sum(),
                                                   random_state=42)])

        y_balanced_test = np.concatenate([np.ones(user_mask_test.sum()),
                                          np.zeros(user_mask_test.sum())])  # 1 for user, 0 for non-user

        # Shuffling the balanced train and test sets
        X_balanced_train, y_balanced_train = shuffle(X_balanced_train, y_balanced_train, random_state=42)
        X_balanced_test, y_balanced_test = shuffle(X_balanced_test, y_balanced_test, random_state=42)

        # Create and compile the model
        model = create_neural_network(X_train.shape[1], binary=True)

        # Train the model with fewer epochs and a larger batch size
        model.fit(X_balanced_train, y_balanced_train, epochs=5, batch_size=64)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_balanced_test, y_balanced_test)

        print('Test loss:', loss)
        print('Test accuracy:', accuracy)

        # Generate and print the confusion matrix
        y_pred = model.predict(X_balanced_test).ravel()

        y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred]
        print(confusion_matrix(y_balanced_test, y_pred_class))

    # unique_users_train = np.unique(y_train)
    # unique_users_test = np.unique(y_test)
    # y_train = label_binarize(y_train, classes=unique_users_train.tolist())
    # y_test = label_binarize(y_test, classes=unique_users_test.tolist())
    #
    # n_classes_train = y_train.shape[1]
    # n_classes_test = y_test.shape[1]
    #
    # # Learn to predict each class against the other using RandomForest
    # classifier = OneVsRestClassifier(MLPClassifier())
    # classifier.fit(X_train, y_train)
    #
    # # Predict the classes
    # y_score = classifier.predict(X_test)
    #
    # print(y_score)
    #
    # # Precision, Recall, F1-score
    # precision = precision_score(y_test, y_score, average='micro')
    # recall = recall_score(y_test, y_score, average='micro')
    # f1 = f1_score(y_test, y_score, average='micro')
    #
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1-score: ", f1)
    #
    # # ROC AUC
    # # For ROC AUC, we need probability estimates of the positive class
    # # The classifier should have predict_proba method
    #
    # y_score_proba = classifier.predict_proba(X_test)
    # roc_auc = roc_auc_score(y_test, y_score_proba, multi_class='ovr', average='micro')
    #
    # print("ROC AUC: ", roc_auc)
