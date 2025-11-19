import logging
import os
import sys
from datetime import datetime
from typing import Optional

_LOGGERS_INITIALIZED = set()

LOG_COLORS = {
    "DEBUG": "\033[37m",  # Gris claro
    "INFO": "\033[36m",  # Cian
    "WARNING": "\033[33m",  # Amarillo
    "ERROR": "\033[31m",  # Rojo
    "CRITICAL": "\033[41m",  # Fondo rojo
}
RESET_COLOR = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter que colorea solo el nivel de log."""

    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{RESET_COLOR}"
        return super().format(record)


def setup_logger(
    name: str = "AmbuHive",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    enable_file_logging: bool = True,
) -> logging.Logger:
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
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level or level)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
        logger.addHandler(file_handler)

    logger.propagate = False

    print(f"Logger '{name}' configurado - Nivel: {logging.getLevelName(level)}")

    return logger


def get_module_logger(module_name: str) -> logging.Logger:
    """Obtiene logger para módulo específico sin configurar handlers nuevos"""
    if module_name == "__main__":
        return logging.getLogger("AmbuHive")

    base_logger = logging.getLogger("AmbuHive")
    module_logger = base_logger.getChild(module_name.split(".")[-1])

    return module_logger


def setup_default_logger():
    """Configura el logger por defecto una sola vez"""
    return setup_logger(
        name="AmbuHive",
        log_dir="./logs",
        level=logging.INFO,
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        enable_file_logging=True,
    )


_root_logger = setup_default_logger()

if __name__ == "__main__":
    # Solo para pruebas - usar el logger ya configurado
    logger = logging.getLogger("AmbuHive")

    logger.debug("Mensaje de debug - detalles técnicos")
    logger.info("Mensaje informativo - operación normal")
    logger.warning("Advertencia - algo inusual pero manejable")
    logger.error("Error - operación fallida")
    logger.critical("Crítico - error grave que requiere atención inmediata")

    module_logger = get_module_logger("map_downloader")
    module_logger.info("Este es un mensaje desde un módulo específico")

    try:
        result = 10 / 0
    except Exception as e:
        logger.exception("Error en operación matemática: %s", e)
