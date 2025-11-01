from py4j.java_gateway import logging


SEED = 42
BIG_PRIME = 4294967377  # A prime number bigger than 2^32 (The maximum shingle hash value computed by mmh3)
SHINGLE_PYSPARK_THRESHOLD = 10_000  # TODO adjust this

logging.basicConfig(level=logging.INFO)
