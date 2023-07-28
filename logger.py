import logging
import sys
import os

import tensorflow

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = os.path.normpath(os.path.join(os.getcwd(), "logs/logs.log"))


def get_console_handler(formattter):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formattter)
    return console_handler


def get_file_handler(log_file: str, formattter):

    if not os.path.isfile(log_file) and os.path.exists(log_file):
        with open('logs.log', 'w+') as f:
            f.write('')
    elif not os.path.exists(log_file):
        raise Exception(f'No such directory: {log_file}')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formattter)
    file_handler.setLevel(logging.ERROR)
    return file_handler


# Add handlers to the logger
# logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, handlers=[get_console_handler(FORMATTER), get_file_handler(LOG_FILE, FORMATTER)])
logger = logging.getLogger(__name__)

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
