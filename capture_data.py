#! python
# listeners

import pynput.keyboard as keyboard
import pynput.mouse as mouse
import system_features_extractor as SFExtractor
from re import search
from nltk.tokenize import RegexpTokenizer
import os.path, system_features_extractor
from PyPDF2 import PdfFileReader
from docx import Document

class Combinations:
    """A class which includes key combinations or sets of keys."""
    END_KEYS = [keyboard.Key.esc, keyboard.Key.f4]  # key combination to finish listening
    NEXT_WORD_KEYS = [keyboard.Key.space, ':', ",", "/", '"']  # key of next words
    NEW_CONTEXT_KEYS = [keyboard.Key.down, keyboard.Key.up]  # , keyboard.Key.dot},
    SENTENCE_END_KEYS = [keyboard.Key.enter, ".", ";", '?', '!']
    NUMPAD_NUMBERS_KEYS = [keyboard.Key.n_zero, keyboard.Key.n_one, keyboard.Key.n_two,
                           keyboard.Key.n_three, keyboard.Key.n_four,
                           keyboard.Key.n_five, keyboard.Key.n_six, keyboard.Key.n_seven,
                           keyboard.Key.n_eight, keyboard.Key.n_nine]

class RealTimeKeyListener:
    __left_button_mouse_is_pressed = False
    __position = 0  # parameter is <=0
    __previous_key = None
    __sentence = ""
    __list_of_words = None
    __keys_counter = {}
    __non_printable_counter = {}
    __non_printable_digraphs = []
    __pressed_keys = []
    destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"
    keyboard_listener = None
    mouse_listener = None

    def __init__(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.__on_press)
        self.mouse_listener = mouse.Listener(on_click=self.__on_click)
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def __on_click(self, x, y, button, pressed):
        self.__count_clicks(button)
        if pressed and button == mouse.Button.left:
            self.__left_button_mouse_is_pressed = True

    def __on_press(self, key):
        """A method which is called whenever user presses a key. It checks type of typing key and call other functions,
         whenever definded trigger happens."""
        self.__count_clicks(key)
        self.__count_no_printable_keys(key)
        SFExtractor.add_list_to_json_file(self.destination_json_file_path, 'Pressed keys', self.__pressed_keys)
        SFExtractor.add_list_to_json_file(self.destination_json_file_path, 'Digraphs', self.__non_printable_digraphs)
        # print('Digraphs: ', self.__non_printable_digraphs)
        if self.__previous_key in Combinations.END_KEYS and key in Combinations.END_KEYS:
            self.__is_finished()

        elif ((key in Combinations.SENTENCE_END_KEYS) or (key in Combinations.NEW_CONTEXT_KEYS)
              or (self.__left_button_mouse_is_pressed) or (
                      hasattr(key, 'char') and key.char in Combinations.SENTENCE_END_KEYS)) \
                and len(self.__sentence) > 0:
            self.__on_finished_context()

        elif key == keyboard.Key.delete and self.__sentence and self.__position != 0:
            self.__delete_chars()

        elif key == keyboard.Key.backspace and self.__sentence:
            self.__backspace_chars()

        elif key == keyboard.Key.left and abs(self.__position) <= len(self.__sentence):
            self.__position -= 1

        elif key == keyboard.Key.right and self.__position < 0:
            self.__position += 1

        elif (hasattr(key, 'char') and key.char is not None and len(key.char) < 2) or (
                key in Combinations.NUMPAD_NUMBERS_KEYS or key == keyboard.Key.space):
            self.__insert_key(key)
        self.__previous_key = key
        self.__left_button_mouse_is_pressed = False
        self.__pressed_keys.clear()
        self.__non_printable_digraphs.clear()

    def __on_finished_context(self, at_the_end=False):
        """ Method that checks add list of words to file whenever the NEW_CONTEXT_KEYS or  combination is entered."""
        if search('[0-9a-zA-Z]', self.__sentence):
            if self.__position == 0:
                self.__list_of_words = SFExtractor.ListOfWords(self.__sentence)
                if self.__list_of_words.all_words:
                    SFExtractor.write_object_to_json_file(self.destination_json_file_path, 'Sentence',
                                                          SFExtractor.object_to_dicts(self.__list_of_words))
                self.__sentence = ''
            else:
                self.__list_of_words = SFExtractor.ListOfWords(self.__sentence[:self.__position])
                if self.__left_button_mouse_is_pressed:
                    self.__list_of_words.set_left_click()
                    self.__list_of_words = None

                SFExtractor.write_object_to_json_file(self.destination_json_file_path, 'Sentence',
                                                      SFExtractor.object_to_dicts(self.__list_of_words))
                if self.__left_button_mouse_is_pressed or at_the_end:
                    self.__list_of_words = SFExtractor.ListOfWords(self.__sentence[self.__position:])
                    if self.__left_button_mouse_is_pressed:
                        self.__list_of_words.set_left_click()
                    if self.__list_of_words.all_words:
                        SFExtractor.write_object_to_json_file(self.destination_json_file_path, 'Sentence',
                                                              SFExtractor.object_to_dicts(self.__list_of_words))

                    self.__list_of_words = None
                    self.__sentence = ''
                    self.__position = 0
                else:
                    self.__sentence = self.__sentence[self.__position:]
        self.__list_of_words = None

    def __is_finished(self):
        """A method that checks if the END KEY COMBINATION is clicked by user """
        SFExtractor.add_simple_dict_to_json_file(self.destination_json_file_path, 'Keys', self.__keys_counter)
        SFExtractor.add_simple_dict_to_json_file(self.destination_json_file_path, 'No printable keys',
                                                 self.__non_printable_counter)
        if self.__sentence:
            self.__on_finished_context(at_the_end=True)
        self.mouse_listener.stop()
        self.keyboard_listener.stop()

    def __count_clicks(self, click):
        """A method that adds each mouse button click to dictionary collecting each click event."""
        if str(click) not in self.__keys_counter.keys():
            self.__keys_counter[str(click)] = 1
        else:
            self.__keys_counter[str(click)] += 1
        self.__record_non_printable_digraphs_keys(click)
        self.__pressed_keys.append(str(click))

    def __record_non_printable_digraphs_keys(self, click):
        if (isinstance(click, keyboard.Key) and click not in Combinations.NUMPAD_NUMBERS_KEYS and self.__pressed_keys is not None) or (isinstance(self.__previous_key, keyboard.Key) and click not in Combinations.NUMPAD_NUMBERS_KEYS):
            self.__non_printable_digraphs.append([str(self.__previous_key), str(click)])

    def __count_no_printable_keys(self, click):
        if isinstance(click, keyboard.Key) and click not in Combinations.NUMPAD_NUMBERS_KEYS:
            if str(click) not in self.__non_printable_counter.keys():
                self.__non_printable_counter[str(click)] = 1
            else:
                self.__non_printable_counter[str(click)] += 1

    def __delete_chars(self):
        """A method which is called whenever user presses 'delete' key to delete chars from current writting sentence"""
        if self.__position < -1:
            self.__sentence = self.__sentence[:self.__position] + self.__sentence[self.__position + 1:]
        elif self.__position == -1:
            self.__sentence = self.__sentence[:-1]
        self.__position += 1

    def __backspace_chars(self):
        """A Method which is called whenever user presses 'backspace' key to delete chars from sentence """
        if self.__position == 0:
            self.__sentence = self.__sentence[:-1]
        elif self.__position < 0:
            self.__sentence = self.__sentence[:self.__position - 1] + self.__sentence[self.__position:]

    def __insert_key(self, key):
        char = ''
        if hasattr(key, 'char') and key.char is not None and len(key.char) < 2:
            char = key.char
        elif key in Combinations.NUMPAD_NUMBERS_KEYS or key == keyboard.Key.space:
            char = key._value_.char
        if char is not None and char.isprintable():
            if self.__position == 0:
                self.__sentence += char
            else:
                self.__sentence = self.__sentence[:self.__position] + char + self.__sentence[self.__position:]


class OfflineListener:
    """ A class which should be used for 'offline' files. """
    # tokenizer to separate new context (mostly sentences)
    __sentence_tokenizer = RegexpTokenizer('[.;!?\n]', gaps=True)
    destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"
    source_txt_file_path = "C:/Users/user/Desktop/destination_file.json"
    file_types = ['txt', 'pdf', 'docx']

    def read_text_file(self) -> str:
        """A method that reads a file .txt, .docx or .pdf and returns text as string. """
        if os.path.isfile(self.source_txt_file_path) and os.path.getsize(self.source_txt_file_path) > 0:
            text = ''
            if self.source_txt_file_path[-3:] == 'txt':
                with open(self.source_txt_file_path) as f:
                    text = f.read()
            elif self.source_txt_file_path[-4:] == 'docx':
                doc = Document(self.source_txt_file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text
            elif self.source_txt_file_path[-3:] == 'pdf':
                pdf_file = open(self.source_txt_file_path, 'rb')
                pdf_reader = PdfFileReader(pdf_file)
                for page_num in range(0, pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)
                    text += page.extractText()
            return text

    def write_to_json_file(self, text):
        """A method that writes input text to json file"""
        for sentence in self.__sentence_tokenizer.tokenize(text):
            __list_of_words = system_features_extractor.ListOfWords(sentence)
            __list_of_words.is_from_file = True
            if __list_of_words.all_words:
                system_features_extractor.write_object_to_json_file(self.destination_json_file_path, 'Sentences',
                                                                    system_features_extractor.object_to_dicts(
                                                                        __list_of_words))
            __list_of_words.clear_list()