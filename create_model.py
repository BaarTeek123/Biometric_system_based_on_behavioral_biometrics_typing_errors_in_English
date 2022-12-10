import math
import random
import sys
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from json import load
import numpy as np
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_classif
from iteration_utilities import random_combination

# assign ids to pos_tags
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'WDT',
            'WP', 'WP$', 'WRB']
tmp = {}
i = 0
for tag in pos_tags:
    tmp[tag] = i
    i += 1
pos_tags = tmp
del tmp

# source_files
directory = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'
file_pth = 'json_files/done_miki.json'

# define users and columns to read
user_names = {}
# program_is_ver_sim = False
program_is_ver_sim = True


# load data
def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = load(f)
    return data


def get_misspelled_words_df_from_json(file_path: str, labeled: bool = True, use_tags: bool = True):
    cols = [
        # edit ops
        'damerau_levenshtein_distance',
        'jaro_winkler_ns',
        # # # # token based
        'gestalt_ns',
        'sorensen_dice_ns',
        'overlap',
        # # phonetic
        'mra_ns',
        # # # seq based
        'lcsstr',
        'ml_type_id',
        'ml_operation_subtype_id',
        'ml_det0',
        'ml_det1',
        'ml_det2',
        'ml_det3',
        'ml_det4',
        'ml_det5'
    ]
    global user_names
    misspelled = pd.DataFrame(columns=cols)
    for dictionary in read_json_file(file_path)['Sentence']:
        distances_ml = []
        if 'misspelled_words' in dictionary.keys() and len(dictionary['misspelled_words']) > 0:
            id = 0
            for misspell in dictionary['misspelled_words']:
                if use_tags:
                    tags = [pos_tags[misspell['pos_tag']], pos_tags[misspell['corrected_word_tag']]]
                if 'distance' in misspell.keys() and 'operations' in misspell['distance'].keys():
                    id += 1
                    for dist in cols:
                        if dist in misspell['distance'].keys() and id < 2:
                            if isinstance(misspell['distance'][dist], float) or isinstance(dist, int):
                                distances_ml.append(misspell['distance'][dist])
                    for op in misspell['distance']['operations']:

                        tmp = [float(k) for k in op['ml_repr']]
                        if not use_tags:
                            misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp], columns=cols)],
                                                   ignore_index=True)
                        else:
                            misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp + tags],
                                                                             columns=cols + ['pos_tag_org',
                                                                                             'pos_tag_corrected'])],
                                                   ignore_index=True)
    if labeled:
        name = read_json_file(file_path)['Name']
        if not user_names:
            user_names[name] = 0
        else:
            if name not in user_names.keys():
                user_names[name] = max(user_names.values()) + 1
        misspelled['user_label'] = [user_names[name] for _ in range(misspelled.shape[0])]
    print(misspelled)
    print(misspelled.columns)

    input()
    return misspelled


def load_data(files_directory: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for file in os.listdir(files_directory):
        df = pd.concat([df, get_misspelled_words_df_from_json(files_directory + '\\' + file, labeled=True)],
                       ignore_index=True)

    # drop None (clear data)
    df = df.dropna().reset_index()
    del df['index']
    return df


def create_data_for_v_a_simulation(correct_user_name: str, data_frame: pd.DataFrame, number_of_samples: int = 10):
    user_data = data_frame[data_frame['label'] == user_names[correct_user_name]]
    sample = user_data.sample(n=number_of_samples, replace=False)
    user_data = user_data[user_data.index.isin(sample.index.tolist()) == False]
    user_data.reindex()
    sample.reindex()
    data_frame = pd.concat([data_frame[data_frame['user_label'] != user_names[correct_user_name]], user_data])
    del user_data
    return data_frame, sample


scaler = StandardScaler()


def choose_features(number_of_features, X, y):
    selector = SelectKBest(f_classif, k=number_of_features)
    X = selector.fit_transform(X, y)
    f_score_column_indexes = (-selector.scores_).argsort()[:number_of_features]  # choosen featuers indexes
    return X, sorted(f_score_column_indexes)


cols = [
    # edit ops
    'damerau_levenshtein_distance',
    'jaro_winkler_ns',
    # # # token based
    'gestalt_ns',
    'sorensen_dice_ns',
    'overlap',
    # # # phonetic
    'mra_ns',
    # # # seq based
    'lcsstr',
    'ml_type_id',
    'ml_operation_subtype_id',
    'ml_det0',
    'ml_det1',
    'ml_det2',
    'ml_det3',
    'ml_det4',
    'ml_det5',
    'pos_tag_org',
    'pos_tag_corrected',
    'user_label']


def create_ngrams(data_frame: pd.DataFrame, test_size_per_user: int = 10, n_gram_size: int = 5,
                  vecs_size_per_user: int = 30,
                  separate_words: bool = program_is_ver_sim, number_of_features: int = 4):
    users_original_data, user_n_grams, result_X, result_Y, = {}, {}, [], []
    tmp_X, tmp_y = np.array([]), []

    for lab in data_frame['user_label'].unique():
        users_original_data[lab] = data_frame[data_frame['user_label'] == lab]
        users_original_data[lab] = scaler.fit_transform(users_original_data[lab].iloc[:, :-1])
        tmp_X = np.append(tmp_X, users_original_data[lab])
        tmp_y = tmp_y + [lab for _ in range(len(users_original_data[lab]))]

    tmp_y = np.array(tmp_y)
    tmp_X = tmp_X.reshape((tmp_y.shape[0], tmp_X.shape[0] // tmp_y.shape[0]))
    tmp_X, features_cols = choose_features(number_of_features, tmp_X, tmp_y)
    del tmp_y, tmp_X
    for lab in data_frame['user_label'].unique():
        users_original_data[lab] = users_original_data[lab][:, features_cols]

    if separate_words:
        original_test_x_data, user_test_n_grams, test_x, test_y = {}, {}, [], []

        # create other group of data to test
        for lab in users_original_data:
            idx = np.random.randint(users_original_data[lab].shape[0], size=test_size_per_user)
            original_test_x_data[lab] = users_original_data[lab][idx, :]
            users_original_data[lab] = np.delete(users_original_data[lab], idx, axis=0)
    num_of_vecs_per_user = math.comb(vecs_size_per_user, program_n_gram_size)
    for u_id in users_original_data.keys():
        user_n_grams[u_id] = []
        tmp_set = set()
        while len(tmp_set) < num_of_vecs_per_user:
            tmp_set.add(random_combination(range(len(users_original_data[u_id])), n_gram_size))
        for n_gram in tmp_set:
            user_n_grams[u_id].append(np.array([users_original_data[u_id][n_gram_el] for n_gram_el in n_gram]))
        user_n_grams[u_id] = np.array(user_n_grams[u_id])
        for vec_id in range(len(user_n_grams[u_id])):
            result_X.append(user_n_grams[u_id][vec_id].flatten())
        for _ in range(num_of_vecs_per_user):
            result_Y.append(u_id)
    if separate_words:
        for u_id in original_test_x_data.keys():
            original_test_x_data[u_id] = np.array(list(combinations(original_test_x_data[u_id], n_gram_size)))
            for vec_id in range(len(original_test_x_data[u_id])):
                test_x.append(original_test_x_data[u_id][vec_id].flatten())
            for _ in range(original_test_x_data[u_id].shape[0]):
                test_y.append(u_id)
        del users_original_data, user_n_grams
        print(np.array(result_X), np.array(result_Y), np.array(test_x), np.array(test_y), features_cols)
        return np.array(result_X), np.array(result_Y), np.array(test_x), np.array(test_y), features_cols
    del users_original_data, user_n_grams
    return np.array(result_X), np.array(result_Y), features_cols


df = load_data(directory)[cols]
minimum_words_num = min ([df[df['user_label'] == k].reset_index(drop=True).shape[0] for k in df['user_label'].unique()])
number_of_features = 7

program_n_gram_size = 4

program_test_size_per_user = int(minimum_words_num * 0.4)
program_num_of_vecs_per_user = int(minimum_words_num * 0.45)

if program_is_ver_sim:
    X, y, X_test, y_test, features_cols = create_ngrams(df, program_test_size_per_user, program_n_gram_size,
                                                        program_num_of_vecs_per_user, program_is_ver_sim,
                                                        number_of_features=number_of_features)

else:
    X, y, features_cols = create_ngrams(df, program_test_size_per_user, program_n_gram_size,
                                        program_num_of_vecs_per_user,
                                        program_is_ver_sim)

if X.shape[0] / (X_test.shape[0] + X.shape[0]) < 0.70:

    idxs = np.random.choice(X_test.shape[0], abs(X_test.shape[0] - X.shape[0] // 5), replace=False)
    X_test = np.delete(X_test, idxs, 0)
    y_test = np.delete(y_test, idxs, 0)
elif X.shape[0] / (X_test.shape[0] + X.shape[0]) > 0.8:
    idxs = np.random.choice(X.shape[0], abs(4*X_test.shape[0] - X.shape[0]), replace=False)
    X = np.delete(X, idxs, 0)
    y = np.delete(y, idxs, 0)
print(X)
# input()
print(X_test)


del df
del scaler
del pos_tags
