import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from classifiers import create_neural_network, build_tuned_nn, build_tuned_rfc, param_grid
from sklearn.neural_network import MLPClassifier
import pandas as pd
from create_model import create_dataset, user_names
from sklearn.svm import SVC
from cv import run_cv
from sklearn.metrics import accuracy_score
import numpy as np

import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    """
    X_train, y_train, X_test, y_test = create_dataset(test_ratio=0)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    print(X_test[:10])
    y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
    y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
    y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))
    print(y_test[:10])
    # add logger
    # force stop if accuracy is going down
    earlystopping = keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  mode="max", patience=7,
                                                  restore_best_weights=True)
    logger = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )
    model = create_neural_network(X_train.shape[1], len(user_names.keys()))
    history = model.fit(tf.expand_dims(X_train, axis=-1), y_train, validation_data=(X_valid, y_valid), epochs=50,
                        batch_size=128, callbacks=[earlystopping, logger])
    # predict
    pred = model.predict(X_test)


    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # X_train, y_train, X_test, y_test = create_dataset(test_ratio=0.3)
    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    # y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
    # y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
    # y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))


    # define classifiers
    # CLASSIFIERS = [
    #     # (build_tuned_rfc,
    #     #  {'clf': RandomForestClassifier(), 'x_train': X_valid, 'y_train': y_valid, 'param_grid': param_grid['Random Forest']},
    #     #  'Random Forest'),
    #
    #     (build_tuned_rfc,
    #      {'clf': KNeighborsClassifier(), 'x_train': X_valid, 'y_train': y_valid, 'param_grid': param_grid['K-Nearest Neighbors']},
    #
    #      'K-Nearest Neighbors'),
    #
    #     (build_tuned_rfc,
    #      {'clf':  SVC(probability=True), 'x_train': X_valid, 'y_train': y_valid, 'param_grid': param_grid['SVC']},
    #      'SVC'),
    #
    #     (build_tuned_rfc,
    #      {'clf': GradientBoostingClassifier(), 'x_train': X_valid, 'y_train': y_valid, 'param_grid': param_grid['Gradient Boosting']},
    #      'Gradient Boosting'),
    #
    #     (build_tuned_rfc,
    #      {'clf': MLPClassifier(), 'x_train': X_valid, 'y_train': y_valid,
    #       'param_grid': param_grid['MLP Classifier']},
    #      'MLP Classifier'),


        # (create_neural_network, {'input_dim': X_train.shape[1]}, 'Neural Network'),
        # (build_tuned_nn, {'x_train': X_train, 'y_train': y_train}, 'Tuned Neural Network'),
    # ]
"""



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
    del X_test, y_test
    results = pd.DataFrame()
    print(f'X_train: {len(X_train)}')
    print(f'X_valid: {len(X_valid)}')
    for clf, params, clf_name in CLASSIFIERS:
        cv_res = run_cv(clf, params, X_train, y_train, X_valid, y_valid, 5, 2, 0.3)
        results = pd.concat([results, cv_res], ignore_index=True)
        cv_res.to_csv(f'{clf_name}.csv', mode='w+')
    results.to_csv('results.csv', mode='w+')
