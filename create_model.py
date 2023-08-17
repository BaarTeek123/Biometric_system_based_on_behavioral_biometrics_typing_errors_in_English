from n_grams_creator import load_data, choose_features, scale_data, create_users_ngrams, create_labeled_test_and_train_buckets
import os
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler


DIRECTORY = os.path.join('models', 'json_files')

NUMBER_OF_FEATURES = 5 # liczba cech / słowo
N_GRAM_SIZE = 3 # liczba słow w wektorze
AMOUNT_OF_N_GRAMS_PERS_USER = 35000

COLUMNS = [
    # edit ops
    'damerau_levenshtein_distance',
    'jaro_winkler_ns',
    # token based
    'gestalt_ns',
    'sorensen_dice_ns',
    'overlap',
    # phonetic
    'mra_ns',
    # seq based
    'lcsstr',

    'ml_type_id',
    'ml_operation_subtype_id',
    'ml_det0',
    'ml_det1',
    'ml_det2',
    'ml_det3',
    'ml_det4',
    'ml_det5']
# 'user_label']

def create_dataset(number_of_features = NUMBER_OF_FEATURES,amount_of_n_grams_pers_user = AMOUNT_OF_N_GRAMS_PERS_USER,
                   n_gram_size = N_GRAM_SIZE, source_directory=DIRECTORY, columns=COLUMNS, verbose_mode=False, test_ratio = 0.5, if_separate_words = True, scaler = Normalizer()):
    # load data from directory
    df = load_data(source_directory, columns)

    # split into 2 sets
    # test_valid_bucket_per_user = int(minimum_words_num * 0.5)
    # train_bucket_per_user = int(minimum_words_num * 0.5)
    # misspelled words level1
    if if_separate_words:
        X_train_bucket, X_test_bucket, y_train_bucket, y_test_bucket = create_labeled_test_and_train_buckets(df, test_ratio)
        # scale data
        X_train_bucket, X_test_bucket = scale_data(X_train_bucket, X_test_bucket, scaler=scaler)
        # transform  data frames to np.arrays
        # y_train_bucket, y_test_bucket = y_train_bucket.to_numpy(), y_test_bucket.to_numpy()
        y_train_bucket, y_test_bucket = y_train_bucket.values.ravel(), y_test_bucket.values.ravel()

        # choose columns
        X_train_bucket, X_test_bucket, chosen_columns_ids = choose_features(number_of_features, f_classif,
                                                                        X_train_bucket, y_train_bucket, X_test_bucket)
        # chosen columns names
        chosen_columns_names = df.columns[chosen_columns_ids]

        X_train, y_train = create_users_ngrams(X_train_bucket, y_train_bucket, amount_of_n_grams_pers_user, n_gram_size)
        del X_train_bucket, y_train_bucket
        X_test, y_test = create_users_ngrams(X_test_bucket, y_test_bucket, amount_of_n_grams_pers_user, n_gram_size)
        del X_test_bucket, y_test_bucket


    else:
        X_train_bucket, X_test_bucket, y_train_bucket, y_test_bucket = create_labeled_test_and_train_buckets(df, 0.0)
        # scale data
        X_train_bucket = scale_data(X_train_bucket, scaler=StandardScaler())
        # transform  data frames to np.arrays

        y_train_bucket = y_train_bucket.values.ravel()

        # choose columns
        X_train_bucket, chosen_columns_ids = choose_features(number_of_features, f_classif,
                                                                            X_train_bucket, y_train_bucket)
        # chosen columns names
        chosen_columns_names = df.columns[chosen_columns_ids]

        X_train, y_train = create_users_ngrams(X_train_bucket, y_train_bucket, amount_of_n_grams_pers_user, n_gram_size)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, test_size=test_ratio)




    # delete data frame
    del df
    if verbose_mode:
        return X_train, y_train, X_test, y_test, chosen_columns_names
    return X_train, y_train, X_test, y_test

