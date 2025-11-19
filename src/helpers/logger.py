import logging
import os
import sys
import threading
from datetime import datetime
from typing import Optional

_LOGGERS_INITIALIZED = set()
_LOGGER_LOCK = threading.Lock()

LOG_COLORS = {
    "DEBUG": "\033[37m",  # Gris claro
    "INFO": "\033[36m",  # Cian
    "WARNING": "\033[33m",  # Amarillo
    "ERROR": "\033[31m",  # Rojo
    "CRITICAL": "\033[41m",  # Fondo rojo
}
RESET_COLOR = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS and sys.stdout.isatty():
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{RESET_COLOR}"
        return super().format(record)


class SingleFileHandler(logging.FileHandler):
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"thoraxscan_{timestamp}.log")

        super().__init__(log_file, encoding="utf-8")


def setup_logger(
    name: str = "ThoraxScan",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = logging.DEBUG,
    enable_file_logging: bool = True,
) -> logging.Logger:
    with _LOGGER_LOCK:
        if name in _LOGGERS_INITIALIZED:
            return logging.getLogger(name)

        _LOGGERS_INITIALIZED.add(name)

        logger = logging.getLogger(name)

        if logger.handlers:
            return logger

        logger.setLevel(level)

        log_format = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        datefmt = "%H:%M:%S"

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level or level)
        console_handler.setFormatter(ColoredFormatter(log_format, datefmt=datefmt))
        logger.addHandler(console_handler)

        if enable_file_logging:
            file_handler = SingleFileHandler(log_dir)
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(log_format, datefmt=datefmt)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

        logger.info(
            f"Logger '{name}' configurado - Nivel: {logging.getLevelName(level)}"
        )
        if enable_file_logging:
            logger.info(f"Archivo de log: {file_handler.baseFilename}")

        return logger


def get_module_logger(module_name: str) -> logging.Logger:
    if module_name == "__main__":
        return logging.getLogger("ThoraxScan")

    base_logger = logging.getLogger("ThoraxScan")
    module_logger = base_logger.getChild(module_name.split(".")[-1])

    return module_logger


_global_logger = setup_logger()


def log_function_call(log_args=True, log_result=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_module_logger(func.__module__)

            # Log de entrada
            if log_args:
                args_str = ", ".join([str(arg) for arg in args[1:]])  # Skip self
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.debug(f"→ {func.__name__}({all_args})")
            else:
                logger.debug(f"→ {func.__name__}()")

            try:
                result = func(*args, **kwargs)
                if log_result:
                    logger.debug(f"← {func.__name__}() = {result}")
                else:
                    logger.debug(f"← {func.__name__}() completado")
                return result
            except Exception as e:
                logger.error(f"✗ {func.__name__}() falló: {str(e)}")
                raise

        return wrapper

    return decorator
