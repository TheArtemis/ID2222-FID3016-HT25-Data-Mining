import logging
import os

import colorlog

# Import TIME level to ensure it's registered (side effect: adds TIME level to logging)
from miner.decorators import TIME_LEVEL  # noqa: F401


SEED = 42
BIG_PRIME = 4294967377  # A prime number bigger than 2^32 (The maximum shingle hash value computed by mmh3)
SHINGLE_PYSPARK_THRESHOLD = 1_000_000  # TODO adjust this

# Configure colorful logging
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "TIME": "purple",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Remove logging we dont like
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("networkx").setLevel(logging.WARNING)
logging.getLogger("scipy").setLevel(logging.WARNING)
logging.getLogger("scikit-learn").setLevel(logging.WARNING)
logging.getLogger("pyspark").setLevel(logging.WARNING)
logging.getLogger("kagglehub").setLevel(logging.WARNING)
logging.getLogger("colorlog").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


# Only print ASCII art if not in a Spark worker process
if os.environ.get("SPARK_ENV_LOADED") is None:
    logger.info("""\n
            ░███     ░███ ░██                                       
     ░██    ░████   ░████                                      ░██  
    ░██     ░██░██ ░██░██ ░██░████████   ░███████  ░██░████     ░██ 
   ░██      ░██ ░████ ░██ ░██░██    ░██ ░██    ░██ ░███          ░██
    ░██     ░██  ░██  ░██ ░██░██    ░██ ░█████████ ░██          ░██ 
     ░██    ░██       ░██ ░██░██    ░██ ░██        ░██         ░██  
            ░██       ░██ ░██░██    ░██  ░███████  ░██              
                                                                    
                                                                    
                                                                    """)
