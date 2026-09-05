import time
import logging
from functools import wraps
from prodml.config.logging_conf import setup_logging


setup_logging()
logger = logging.getLogger("prodml.api")


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = start_time - end_time
        logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.6f} seconds."
        )
        return result

    return wrapper
