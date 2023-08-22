import tensorflow as tf
from n_grams_creator import user_names
from keras import Sequential
from keras import layers
from keras import optimizers
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import RandomizedSearchCV
from keras import initializers
print(tf.config.list_physical_devices('GPU'))
from keras import callbacks


# def create_neural_network(input_dim, output_dim=len(user_names.keys()), binary=False):
#     # create sequential model
#     model = keras.Sequential()
#     model.add(keras.layers.Dense(128, activation='relu', name='layer_1', input_dim=input_dim))
#     model.add(keras.layers.BatchNormalization())
#     # model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(64, activation='relu', name='layer_2'))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(32, activation='relu', name='layer_3'))
#     model.add(keras.layers.BatchNormalization())
#     # model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(16, activation='relu', name='layer_4'))
#     model.add(keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
#                                               beta_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
#                                               gamma_initializer=keras.initializers.Constant(value=0.9)))
#     model.add(keras.layers.Dropout(0.5))
#     if not binary:
#         model.add(keras.layers.Dense(output_dim, activation='softmax', name='output_layer'))
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     else:
#         model.add(
#             keras.layers.Dense(1, activation='sigmoid', name='output_layer'))  # Two classes: the user and all others
#
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Binary problem now
#
#     return model

earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                                        mode="max", patience=7,
                                        restore_best_weights=True)
logger = callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )
def create_neural_network(input_dim,output_dim, binary=False):
    # create sequential model
    model = Sequential()
    model.add(layers.Dense(128, activation='relu', name='layer_1', input_dim=input_dim))
    model.add(layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', name='layer_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu', name='layer_3'))
    model.add(layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', name='layer_4'))
    model.add(layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                        beta_initializer=initializers.RandomNormal(mean=0.0,
                                                                                   stddev=0.05),
                                        gamma_initializer=initializers.Constant(value=0.9)))
    if not binary:
        model.add(layers.Dense(output_dim, activation='softmax', name='output_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(layers.Dense(1, activation='sigmoid',
                               name='output_layer'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model_nn(hp, output_dim=len(user_names.keys())):
    model = Sequential()
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(output_dim, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_tuned_nn(x_train, y_train):
    tuner = RandomSearch(build_model_nn, objective='val_accuracy',
                         max_trials=5, overwrite=True, directory='./project')
    # Perform hyperparameter search
    tuner.search(x_train, y_train, epochs=100, validation_split=0.15)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return tuner.hypermodel.build(best_hps)


param_grid = {
    'Random Forest': {
        'n_estimators': [10, 50, 100],  # , 100, 200],
        # 'max_depth': [None, 10, 20, 30],  # , 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['log2', 'sqrt', None]
    },
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 11],
                            'weights': ['uniform', 'distance'],
                            'metric': ['euclidean', 'manhattan']
                            },
    'Gradient Boosting': {'n_estimators': [100, 200],
                          'learning_rate': [0.01, 0.1, 1],
                          'max_depth': [3, 10],  # , 20, None],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 4],
                          'max_features': ['sqrt', 'log2', None]
                          },
    'SVC': {'kernel': ['rbf', 'linear'],
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001]
            },
    'MLP Classifier': {"hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
                       "activation": ['tanh', 'relu'],
                       "solver": ['sgd', 'adam'],
                       "alpha": [0.0001, 0.05],
                       "learning_rate": ['constant', 'adaptive'],
                       }
}


def build_tuned_rfc(clf, x_train, y_train, param_grid):
    print(clf)
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=10, cv=2, n_jobs=-1,
                                   random_state=42)
    rf_random.fit(x_train, y_train)
    best_params = rf_random.best_params_
    print(best_params)
    return rf_random.best_estimator_


# if __name__ == "__main__":
# X_train, y_train, X_test, y_test = create_dataset()
# X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
# print(X_test[:10])
# y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
# y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
# y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))
# print(y_test[:10])
# add logger
# force stop if accuracy is going down
# earlystopping = keras.callbacks.EarlyStopping(monitor="val_accuracy",
#                                               mode="max", patience=7,
#                                               restore_best_weights=True)
# logger = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )

# history = model.fit(tf.expand_dims(X_train, axis=-1), y_train, validation_data=(X_valid, y_valid), epochs=50,
#                     batch_size=128, callbacks=[earlystopping, logger])
# # predict
# pred = model.predict(X_test)
#
#
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# threshold = 0.5


"""
    y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
    y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
    y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))

    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
    print(X_test.shape[0] / (X_test.shape[0] + X_train.shape[0]) * 100)
    print(y_train.shape)
    print(y_valid.shape)
    print(y_test.shape)
    print(y_test.shape[0] / y_train.shape[0] * 100)
    # add logger
    logger = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )

    # create sequential model
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', name='layer_1', input_dim=X_train.shape[1]))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu', name='layer_2'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu', name='layer_3'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, activation='relu', name='layer_4'))
    model.add(keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                              beta_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                                              gamma_initializer=keras.initializers.Constant(value=0.9)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(user_names.keys()), activation='softmax', name='output_layer'))
    # add optimizer
    sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # force stop if accuracy is going down
    earlystopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                            mode="max", patience=7,
                                            restore_best_weights=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print("Compiled in", time.time() - start)
    # train model
    history = model.fit(expand_dims(X_train, axis=-1), y_train, validation_data=(X_valid, y_valid), epochs=50,
                        batch_size=128, callbacks=[earlystopping, logger])
    # predict
    pred = model.predict(X_test)
    print("Predicted in", time.time() - start)
    # threshold = 0.5
    # indexes_to_delete = []
    # pred_rejected = []
    # y_deleted = []
    # if len(pred) == len(y_test):
    #     for i in range(len(pred)):
    #         if pred[i].max() < threshold:
    #             pred_rejected.append(pred[i])
    #             y_deleted.append(y_test[i])
    #             indexes_to_delete.append(i)
    # for i in range(len(indexes_to_delete)):
    #     y_test = np.delete(y_test, indexes_to_delete[i] - i, 0)
    #     pred = np.delete(pred, indexes_to_delete[i] - i, 0)
    # del indexes_to_delete

    title = f"using {program_n_gram_size}-graphs."
    file_title = f"{len(user_names.keys())}_{program_n_gram_size}_gram_{len(features_cols)}features"
    models_directory = f"models\\{program_n_gram_size}_gram_{len(features_cols)}features_{len(user_names.keys())}_users"
    graph_directory = f"graphs\\"

    if not path.exists(models_directory):
        makedirs(models_directory)

    plot_result_nn(history, file_title=models_directory + '\\data_loss' + file_title)

    far, frr, eer, threshold = calculate_far_frr_eer(y_test, pred)

    # save results with visualisations
    # 1 Confusion matrix
    plot_confusion_metrics(y_test, pred, list(sorted(user_names.values())),
                           ["user " + str(i + 1) for i in range(len(user_names.values()))],
                           file_title=models_directory + '\\cm' + file_title)
    # 2 ROC curve (system)
    draw_classes_roc_curve(y_test, pred, plot_title="Receiver operating characteristic " + title, classes=user_names,
                           file_title=models_directory + '\\classes_roc_curve_' + file_title, print_micro_macro=None)
    # 3 ROC curve - micro / macro average
    draw_classes_roc_curve(y_test, pred, plot_title="Receiver operating characteristic " + title, classes=user_names,
                           file_title=models_directory + '\\classes_micro_macro_roc_curve_' + file_title)

    # 4 FAR & FRR
    draw_far_frr(far, frr, eer, threshold, file_title=models_directory + '\\far_frr_' + file_title)

    # 5 T-ROC
    draw_system_t_roc_curve(far, frr, eer, plot_title="Receiver operating characteristic " + title,
                            file_title=models_directory + '\\t_roc_' + file_title)
    # 6
    draw_system_roc_curve(far, frr, eer, plot_title="Receiver operating characteristic " + title,
                          file_title=models_directory + '\\roc_' + file_title)
    # 7
    draw_system_roc_curve([1.0] + far, [0.0] + frr, eer, plot_title="Receiver operating characteristic " + title,
                          file_title=models_directory + '\\modified_roc_' + file_title)

    # 1
    copyfile(models_directory + '\\classes_roc_curve_' + file_title + '.png',
             graph_directory + 'classes\\' + file_title + '.png')
    # 2
    copyfile(models_directory + '\\cm' + file_title + '.png', graph_directory + 'cm\\' + file_title + '.png')
    # 3
    copyfile(models_directory + '\\t_roc_' + file_title + '.png', graph_directory + 't_roc\\' + file_title + '.png')
    # 4
    copyfile(models_directory + '\\roc_' + file_title + '.png', graph_directory + 'roc\\' + file_title + '.png')
    # 5
    copyfile(models_directory + '\\modified_roc_' + file_title + '.png',
             graph_directory + 'modified_roc\\' + file_title + '.png')
    # 6
    copyfile(models_directory + '\\far_frr_' + file_title + '.png', graph_directory + 'far_frr\\' + file_title + '.png')

    copyfile(models_directory + '\\classes_micro_macro_roc_curve_' + file_title + '.png',
             graph_directory + 'marco_mircro_classes\\' + file_title + '.png')

    copyfile(models_directory + '\\data_loss' + file_title + '.png',
             graph_directory + 'data_loss\\' + file_title + '.png')
    tn, fp, fn, tp = hipotese_tests(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))

    # with open(directory + '\\pred.txt', 'a+') as f:
    #     f.truncate(0)
    #     for prediction in pred:
    #         f.write(str(prediction) + ",")
    #
    # with open(directory + '\\y_test.txt', 'a+') as f:
    #     f.truncate(0)
    #     for y in y_test:
    #         f.write(str(y) + ",")

    # save model to file
    model.save(models_directory + '\\my_model')

    # measure time
    print(time.time() - start)

    # write to csv file
    with open(models_directory + '\\readme.txt', 'a+') as f:
        f.truncate(0)
        f.write("Features: " + get_info_readme(features_cols))
        f.write(2 * "\n")
        f.write("X_train: " + str(X_train.shape[0]) + "(" + str(round(100 *
                                                                      X_train.shape[0] / (
                                                                              X_train.shape[0] + X_valid.shape[0] +
                                                                              X_test.shape[0]), 2)) + " %)")
        f.write("\nX_valid: " + str(X_valid.shape[0]) + "(" + str(round(100 *
                                                                        X_valid.shape[0] / (
                                                                                X_train.shape[0] + X_valid.shape[0] +
                                                                                X_test.shape[0]), 2)) + " %)")
        f.write("\nX_test: " + str(X_test.shape[0]) + "(" + str(
            round(100 * X_test.shape[0] / (X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]), 2)) + " %)")
        f.write(2 * "\n")
        f.write("\nTP: " + str(tp) + "(" + str(round(tp / (tn + fp + fn + tp), 2)) + " %)")
        f.write("\nTN: " + str(tn) + "(" + str(round(tn / (tn + fp + fn + tp), 2)) + " %)")
        f.write("\nFP: " + str(fp) + "(" + str(round(fp / (tn + fp + fn + tp), 2)) + " %)")
        f.write("\nFN: " + str(fn) + "(" + str(round(fn / (tn + fp + fn + tp), 2)) + " %)")
        f.write(2 * "\n")
        # f.write("\nFail to enrol: " + str(len(pred_rejected)) + "(" + str(100*len(pred_rejected)/(len(pred_rejected) + pred.shape[0])) +"%)")

    columns = ['Name', 'Features', 'N', 'k', 'X_train', 'X_test', 'TN', 'FP', 'FN', 'TP', 'time']
    values = [file_title, program_n_gram_size,str(len(features_cols)), str(X_train.shape[1]), str(X_train.shape[0]), str(X_test.shape[0]),
              str(tn), str(fp), str(fn), str(tp), str(time.time() - start)]
    dictionary = {columns[k]: values[k] for k in range(len(values))}

    save_to_csv("C:\\Users\\user\\PycharmProjects\\bio_system\\results.csv", dictionary)"""
