# File: ui/utils/colored_formatter.py
import logging

COLORS = {
    'DEBUG':    '\033[90m',
    'INFO':     '\033[32m',
    'WARNING':  '\033[33m',
    'ERROR':    '\033[31m',
    'CRITICAL': '\033[41m',
}
WHITE = '\033[97m'
RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    """
    Formatter que colorea el LEVELNAME y fuerza el mensaje en blanco.
    """
    def format(self, record):
        # colorear solo el LEVELNAME
        lvl = record.levelname
        if lvl in COLORS:
            record.levelname = COLORS[lvl] + lvl + RESET

        # envolver el mensaje en blanco
        record.msg = WHITE + record.getMessage() + RESET
        record.args = ()  # ya lo hemos sustituido

        return super().format(record)
