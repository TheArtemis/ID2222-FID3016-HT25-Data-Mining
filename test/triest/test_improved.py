import logging
from pathlib import Path
import matplotlib.pyplot as plt

from miner.core.triest.triest import TriestImproved

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "triest" / "facebook_combined.txt"
logger = logging.getLogger(__name__)

# size of the memory; it could be modified
M = 10000


def test_improved(M):
    M_factors = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    M_values = sorted({max(1, int(M * f)) for f in M_factors})

    estimates_by_M = []
    for m in M_values:
        logger.info(f"Running TriestImproved with M={m}...")
        triest_m = TriestImproved(m)
        est = triest_m.run(DATASET_PATH)
        if callable(est):
            est = est()
        estimates_by_M.append(est)
        logger.info(f"M={m} -> estimate={est}")

    plt.figure(figsize=(8, 5))
    plt.plot(M_values, estimates_by_M, marker="o", linestyle="-")
    plt.xscale("log")
    plt.xlabel("Memory size M")
    plt.ylabel("Triangle estimate")
    plt.title("TriestImproved: estimate vs memory size M")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_improved(M)
