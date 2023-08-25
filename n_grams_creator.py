import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from json import load
import numpy as np
from sklearn.feature_selection import SelectKBest
from decorators import log_info
from logger import logger
from itertools import  combinations
from sklearn.model_selection import train_test_split
import math

# assign ids to pos_tags
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'WDT', 'WP', 'WP$', 'WRB', 'X']


def get_pos_tag(k, pos_tags_list = pos_tags):
    if k in pos_tags_list:
        return pos_tags_list[k]
    else: return 'X'



tmp = {}
i = 0
for tag in pos_tags:
    tmp[tag] = i
    i += 1
pos_tags = tmp
del tmp

# define users and columns to read
user_names = {}


# load data
def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = load(f)
    return data


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

@log_info
def get_misspelled_words_df_from_json(file_path: str, columns: list, use_tags: bool = True, verbose_mode: bool= False):
    cols = columns
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
    # add label
    name = read_json_file(file_path)['Name']
    if not user_names:
        user_names[name] = 0
    else:
        if name not in user_names.keys():
            user_names[name] = max(user_names.values()) + 1
    if verbose_mode:
        misspelled['user_label'] = [name for _ in range(misspelled.shape[0])]
    else:
        misspelled['user_label'] = [user_names[name] for _ in range(misspelled.shape[0])]

    return misspelled


@log_info
def load_data(files_directory: str, columns: list, verbose_mode: bool= False) -> pd.DataFrame:
    df = pd.DataFrame()
    for file in os.listdir(files_directory):
        print(file)
        if file[-4:] == 'json':
            print(file)
            df = pd.concat([df, get_misspelled_words_df_from_json(os.path.normpath(os.path.join(files_directory, file)), verbose_mode=verbose_mode,
                                                                  columns=columns)], ignore_index=True)

    # drop None (clear data)
    df = df.dropna().reset_index()
    del df['index']
    return df


@log_info
def choose_features(number_of_features, foo, X_train, y_train, X_test=None):
    """Function that chooses k best features"""
    if number_of_features <= X_train.shape[1]:
        selector = SelectKBest(foo, k=number_of_features)

    else:
        selector = SelectKBest(foo, k='all')
        logger.warn('K is too high (K > number of columns). All features are selected. ')
    selector.fit(X_train, y_train)
    f_score_column_indexes = (-selector.scores_).argsort()[:number_of_features]  # choosen featuers indexes
    if X_test is not None:
        return selector.transform(X_train), selector.transform(X_test), sorted(f_score_column_indexes)
    return X_train, sorted(f_score_column_indexes)


@log_info
def create_labeled_test_and_train_buckets(data_frame: pd.DataFrame, test_bucket_size: float = 0.5):
    """Function that randomly choose data for trainig and testing to create separated buckets for n-grams input."""

    if isinstance(test_bucket_size, int) and test_bucket_size > 1:
        test_bucket_size = float(test_bucket_size / 100)
    return train_test_split(data_frame.iloc[:, :-1], data_frame.iloc[:, -1], test_size=test_bucket_size, random_state=42)


@log_info
def scale_data(X_train, X_test=None, scaler = StandardScaler()):
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
        return X_train, X_test
    else:
        return X_train


@log_info
def _create_an_user_ngram(list_of_an_user_word_representation: pd.DataFrame, amount_of_ngrams: int, n: int) -> np.array:
    """ Function that create n-grams for a label (user). """
    max_possible_combinations = math.comb(len(list_of_an_user_word_representation.index), n)


    if amount_of_ngrams > max_possible_combinations:
        logger.exception(f"Requested {amount_of_ngrams} combinations, but only {max_possible_combinations} are possible.")
        n_grams_idxs = list(combinations(list(list_of_an_user_word_representation.index), n))




    else:
        n_grams_idxs = set()
        while len(n_grams_idxs) < amount_of_ngrams:
            sample = tuple(sorted(random.sample(list(list_of_an_user_word_representation.index), n)))
            n_grams_idxs.add(sample)

        sequence = (0, 1)

        # Count how many tuples start with the sequence
        print(sum(1 for t in n_grams_idxs if t[:len(sequence)] == sequence))
    tmp_len = len(n_grams_idxs)
    n_grams_idxs = [lst for lst in n_grams_idxs if len(lst) == len(set(lst))]
    logger.info(f"Dropped {tmp_len - len(n_grams_idxs)} duplcates.")
    return np.array([np.take(list_of_an_user_word_representation, idx, axis=0).to_numpy().flatten() for idx in n_grams_idxs])


@log_info

def create_users_ngrams(list_of_word_representation: np.array, list_of_labels: np.array,
                        amount_of_ngrams_per_user: int, n: int):
    """ Function that create n-grams for each label (user)."""
    if len(list_of_word_representation) != len(list_of_labels):
        logger.exception('Size of labels is not equal to the size of the word representations list.')
        raise Exception('Size of labels is not equal to the size of the word representations list.')

    tmp_df = pd.concat(
        [pd.DataFrame(list_of_word_representation), pd.DataFrame(list_of_labels, columns=['user_label'])], axis=1)

    list_of_word_representation, list_of_labels = None, []

    for k in tmp_df['user_label'].unique():

        # create user n_grams with fixed n and amount of n_grams
        print(k, user_names)
        n_grams = _create_an_user_ngram(tmp_df[tmp_df['user_label'] == k].iloc[:, :-1].reset_index(drop=True), amount_of_ngrams_per_user, n)
        if list_of_word_representation is not None:
            list_of_word_representation = np.concatenate([list_of_word_representation, n_grams])
        else:
            list_of_word_representation = n_grams
        list_of_labels = list_of_labels + [k for _ in range(len(n_grams))]
    logger.info(
        f'Created {len(list_of_word_representation)} n-grams ({len(list_of_word_representation) // len(tmp_df["user_label"].unique())} per user)')
    return list_of_word_representation, np.array(list_of_labels)
