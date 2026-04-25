import logging
import os
import sys


def setup_logger(name: str = "reprodbench") -> logging.Logger:
    """
    Create and configure a logger based on .env settings.

    Environment variables:
      LOG_ENABLED   = true | false
      LOG_LEVEL     = DEBUG | INFO | WARNING | ERROR
      LOG_TO_FILE   = true | false
      LOG_FILE_PATH = path/to/logfile.log
    """
    log_enabled = os.getenv("LOG_ENABLED", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_file_path = os.getenv("LOG_FILE_PATH", "reprodbench.log")

    logger = logging.getLogger(name)

    if not log_enabled:
        logger.disabled = True
        return logger

    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_step(logger: logging.Logger, msg: str) -> None:
    """
    Log a pipeline step boundary.
    Example: [STEP] EXECUTION LOOP: starting attempts
    """
    if logger.disabled:
        return
    logger.info(f"[STEP] {msg}")


def log_section(logger: logging.Logger, title: str, fields: dict) -> None:
    """
    Log a titled key-value block.

    Example:
      [EXEC RESULT]
        exit_code: 0
        runtime: 123
    """
    if logger.disabled:
        return

    logger.info(f"[{title}]")
    for k, v in fields.items():
        logger.info(f"  {k}: {v}")


__all__ = ["setup_logger", "log_step", "log_section"]
