from functools import wraps
from time import perf_counter
import logger

def log_info(function):
    def wrapper(*args, **kwargs):
        logger.logger.debug(f"----- {function.__name__}: start -----")
        start = perf_counter()
        output = function(*args, **kwargs)
        logger.logger.debug(f"----- {function.__name__}: finished in {perf_counter()- start:.2f} seconds)-----")
        return output
    return wrapper

def repeat(number_of_times):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(number_of_times):
                func(*args, **kwargs)
        return wrapper
    return decorate