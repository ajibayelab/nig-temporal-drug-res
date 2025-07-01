from logging import Logger


def setup_logger(name: str) -> Logger:

    import logging as log
    import sys

    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            log.FileHandler(f"{name}.log"),
            log.StreamHandler(sys.stdout),
        ],
    )

    return log.getLogger(name)
