# spell checker
import difflib
import os.path
from language_tool_python import LanguageTool
from autocorrect import Speller
import json
from nltk import pos_tag as nltk_pos_tag
from nltk.tokenize import RegexpTokenizer
from re import sub
from string_metrics import Distances


def correct_spelling_autocorrect(sentence) -> str:
    """ A function which returns corrected spelling (by Speller from autocorrect)"""
    return Speller()(sentence)


language_tool = LanguageTool('en-US')


def correct_language_tool(sentence: str) -> str:
    """ A function which returns corrected sentence (by language_tool from language_tool_python)"""
    return language_tool.correct(sentence)


class Word:
    """ A class which includes words, which are separated by NEXT_WORD_KEYS"""

    def __init__(self, word: str, pos_tag: str, corrected_word: str = None, corrected_word_tag: str = None):
        self.original_word = word
        self.pos_tag = pos_tag
        self.corrected_word = corrected_word
        self.corrected_word_tag = corrected_word_tag
        if corrected_word is not None:
            self.distance = Distances(self.original_word, self.corrected_word, is_tokenized=True)
            if not self.distance.__dict__:
                delattr(self, 'distance')


class ListOfWords:
    """ A class which includes words, which are separated by NEXT_WORD_COMBINATION"""
    sentence_tokenizer = RegexpTokenizer('[^(\'\-)\w]', gaps=True)
    __sentence_reg_pattern = r'^([ \W_]*[^A-z])'

    def __init__(self, sentence: str, add_by_left_click: bool = False, is_from_file: bool = False):
        # assign sentence without punctuation
        self.original_sentence = sub(self.__sentence_reg_pattern, '', sentence)
        self.corrected_sentence = correct_language_tool(self.original_sentence)
        # capitalize sentence (due to specificity of the language_tool)
        if self.original_sentence and (
                self.original_sentence[0].islower() and self.corrected_sentence[0].isupper()):
            self.original_sentence = self.original_sentence[:1].upper() + self.original_sentence[1:]
        self.all_words = []
        self.correct_words = []
        self.misspelled_words = []
        self.tokenized_original_sentence = self.sentence_tokenizer.tokenize(self.original_sentence)
        self.tokenized_corrected_sentence = self.sentence_tokenizer.tokenize(self.corrected_sentence)
        # assign corrected_words
        self.original_sentence_structure = [tag[1] for tag in nltk_pos_tag(self.tokenized_original_sentence)]
        if self.original_sentence != self.corrected_sentence:
            self.corrected_sentence_structure = [tag[1] for tag in nltk_pos_tag(self.tokenized_corrected_sentence)]
            if self.corrected_sentence_structure == self.original_sentence_structure:
                self.corrected_sentence_structure = None
                self.sentence_distances = Distances(self.original_sentence, self.corrected_sentence)
        else:
            self.corrected_sentence_structure = None
            self.sentence_distances = None
        i = 0
        if len(self.tokenized_original_sentence) == len(self.tokenized_corrected_sentence):
            for original_word, correct_word in zip(self.tokenized_original_sentence, self.tokenized_corrected_sentence):
                tag = self.original_sentence_structure[i]
                if original_word != correct_word:
                    if self.corrected_sentence_structure is not None:
                        correct_word_tag = self.corrected_sentence_structure[i]
                    else:
                        correct_word_tag = tag
                    self.all_words.append(Word(original_word, tag, correct_word, correct_word_tag))
                    self.misspelled_words.append(Word(original_word, tag, correct_word, correct_word_tag))
                else:
                    self.all_words.append(Word(original_word, tag))
                    self.correct_words.append(Word(original_word, tag))
                i += 1
        else:
            for original_word in self.tokenized_original_sentence:
                tag = self.original_sentence_structure[i]
                if self.tokenized_corrected_sentence:
                    correct_word = difflib.get_close_matches(original_word, self.tokenized_corrected_sentence, n=1)
                    if correct_word:
                        correct_word = str(correct_word[0])
                    else:
                        correct_word = None
                else:
                    correct_word = None
                if original_word != correct_word:
                    if self.corrected_sentence_structure is not None and correct_word:
                        correct_word_tag = self.corrected_sentence_structure[
                            self.tokenized_corrected_sentence.index(correct_word)]
                    else:
                        correct_word_tag = tag
                    self.all_words.append(Word(original_word, tag, correct_word, correct_word_tag))
                    self.misspelled_words.append(Word(original_word, tag, correct_word, correct_word_tag))
                else:
                    self.all_words.append(Word(original_word, tag))
                    self.correct_words.append(Word(original_word, tag))
                i += 1
        self.add_by_left_click = add_by_left_click
        self.is_from_file = is_from_file

    def set_left_click(self):
        self.add_by_left_click = True

    def clear_list(self):
        self.add_by_left_click = False
        self.all_words.clear()
        self.misspelled_words.clear()
        self.correct_words.clear()
        self.original_sentence = None
        self.corrected_sentence = None
        self.is_from_file = None


def get_freq_word(list_of_words: ListOfWords, freq_dict: dict):
    if isinstance(list_of_words, ListOfWords) and isinstance(freq_dict, dict):
        for word_istance in list_of_words.all_words:
            if word_istance.original_word not in freq_dict.keys():
                freq_dict[word_istance.original_word] = 1
            else:
                freq_dict[word_istance.original_word] += 1
    return freq_dict


def object_to_dicts(objct):
    if isinstance(objct, dict):
        return {k: object_to_dicts(v) for k, v in objct.items()}
    elif not isinstance(objct, str) and hasattr(objct, "__iter__"):
        return [object_to_dicts(v) for v in objct]
    elif hasattr(objct, "_ast"):
        return object_to_dicts(objct._ast())
    elif hasattr(objct, "__dict__"):
        return {
            key: object_to_dicts(value)
            for key, value in objct.__dict__.items()
            if not callable(value) and not key.startswith('_')
        }
    else:
        return objct


def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data


def write_object_to_json_file(path_to_file: str, key: str, main_dictionary: dict):
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        # open file
        data = read_json_file(path_to_file)
        # clear file
        open(path_to_file, 'w').close()
        # add data
        file = open(path_to_file, 'a+')
        data[key].append(main_dictionary)
        file.seek(0)
        json.dump(data, file, indent=4)
    else:
        file = open(path_to_file, 'w+')
        tmp = {key: [main_dictionary]}
        json.dump(tmp, file, indent=4)
    file.close()


def add_simple_dict_to_json_file(path_to_file: str, key: str, dict_obj: dict):
    # check if is empty
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        data = read_json_file(path_to_file)
        open(path_to_file, 'w').close()
        file = open(path_to_file, 'a+')
        if isinstance(dict_obj, dict) and dict_obj.keys():
            if key in data.keys():
                for k in dict_obj.keys():
                    if k not in data[key].keys():
                        data[key][k] = dict_obj[k]
                    elif k in data[key].keys() and (isinstance(data[key][k], int) or isinstance(data[key][k], float)):
                        data[key][k] += dict_obj[k]
            else:
                data[key] = dict_obj
        json.dump(data, file, indent=4)
        file.close()


def add_str_to_json(path_to_file: str, user_name: str):
    # check if is empty
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        data = read_json_file(path_to_file)
        open(path_to_file, 'w').close()
        file = open(path_to_file, 'a+')
        data["Name"] = user_name
        json.dump(data, file, indent=4)
        file.close()


def add_list_to_json_file(path_to_file: str, key: str, list_obj: list):
    # check if is empty
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        data = read_json_file(path_to_file)
        open(path_to_file, 'w').close()
        file = open(path_to_file, 'a+')
        if key in data.keys() and isinstance(data[key], list):
            data[key].extend(list_obj)
        elif key not in data.keys():
            data[key] = list_obj
        json.dump(data, file, indent=4)
        file.close()


def extract_data(source_file_path, dest_file_path, user_name: str):
    keys = read_json_file(source_file_path).keys()
    sentences = read_json_file(source_file_path)['Sentence']
    for sentence in sentences:
        if sentence is not None and isinstance(sentence['original_sentence'], str):
            list_of_words = ListOfWords(sentence['original_sentence'], sentence['add_by_left_click'],
                                        sentence['is_from_file'])
            write_object_to_json_file(dest_file_path, 'Sentence', object_to_dicts(list_of_words))
    add_str_to_json(dest_file_path, user_name)
    if 'Keys' in keys:
        add_simple_dict_to_json_file(dest_file_path, 'Keys', read_json_file(source_file_path)['Keys'])
    if 'No printable keys' in keys:
        add_simple_dict_to_json_file(dest_file_path, 'No printable keys',
                                     read_json_file(source_file_path)['No printable keys'])
    if 'Pressed keys' in keys:
        add_list_to_json_file(dest_file_path, 'Pressed keys', read_json_file(source_file_path)['Pressed keys'])
    if 'Digraphs' in keys:
        add_list_to_json_file(dest_file_path, 'Digraphs', read_json_file(source_file_path)['Digraphs'])
