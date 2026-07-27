import logging

from aliases import *
from utils.io_utils import get_log_dir, get_now_str


def setup_logger(
    name: str, level: int = logging.INFO, custom_handle: path_t = None
) -> logging.Logger:
    """Instantiate and configure a logger given a module name and (optionally)
    some configuration options like the logging level.

    Parameters
    ----------
    name: str
        name for the logger; probably you want to use __name__ if you don't have a
        good reason to do otherwise.
    level: int {10, 20, 30}
        integer-valued log level in set {10, 20, 30} (you can use the logging.{INFO, WARN, DEBUG}
        aliases). (default: 20/logging.INFO)
    custom_handle: path_t
        optional path to provide for the file handler (default: uses auto-generated `get_log_dir` to
        write the logs).

    Note
    ----
    `setup_log` works across modules without creating multiple loggers, and with both
    the console stream and a file handler for writing logging messages out; so you can
    call this method in multiple modules/functions and this function handles the I/O
    synchronization.
    """
    # --- create the entry point logger
    logger = logging.getLogger(name)

    if not getattr(logger, "handler_set", None):
        # --- add the file handler
        log_file: path_t = (
            custom_handle
            if custom_handle is not None
            else os.path.join(get_log_dir(), get_now_str() + ".out")
        )
        file_handler = logging.FileHandler(log_file)

        # --- format the file handler
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(fmt)

        # --- setup stream handler
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(fmt)

        # --- configure the logger
        logger.addHandler(file_handler)
        logger.addHandler(console)
        logger.setLevel(level)

        # --- don't add more handlers next time
        logger.handler_set = True
        logger.propagate = False

    return logger
