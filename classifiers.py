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
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import keras_tuner as kt


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


# def build_GradientBoostedTreesModel(hp):
#     model = tfdf.keras.GradientBoostedTreesModel(
#         num_trees=hp.Int('num_trees', 50, 150, step=50),
#         max_depth=hp.Int('max_depth', 4, 8, step=2),
#         # Add other hyperparameters as needed
#     )
#     return model
#
# tuner = kt.Hyperband(
#     build_GradientBoostedTreesModel,
#     objective='val_accuracy',
#     max_epochs=10,
#     factor=3,
#     directory='output_dir',
#     project_name='tfdf_tuning'
# )