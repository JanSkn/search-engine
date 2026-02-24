import logging


def setup_logging(level: str = "DEBUG") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s - %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)
