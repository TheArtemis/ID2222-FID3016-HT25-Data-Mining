"""TRIEST-base and TRIEST-improved implementation (insertion-only streams).

Saved as /mnt/data/triest_impl.py
"""

from collections import defaultdict
import random
from typing import Set, FrozenSet, DefaultDict, Optional, Tuple
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

Edge = FrozenSet[int]

def make_edge(u: int, v: int) -> Edge:
    if u == v:
        raise ValueError("No self-loops supported")
    return frozenset((int(u), int(v)))

class TriestBase:
    def __init__(self, M: int):
        if M < 3:
            raise ValueError("M must be at least 3 for triangle estimates")
        self.M = int(M)
        self.reset()

    def reset(self):
        self.t = 0
        self.S: Set[Edge] = set()
        self.neighbors: DefaultDict[int, Set[int]] = defaultdict(set)
        self.tau = 0.0
        self.tau_vertices: DefaultDict[int, float] = defaultdict(float)

    @property
    def xi(self) -> float:
        if self.t < 3:
            return 1.0
        numer = self.t * (self.t - 1) * (self.t - 2)
        denom = self.M * (self.M - 1) * (self.M - 2)
        if denom == 0:
            return 1.0
        return max(1.0, numer / denom)

    def _add_edge_to_sample(self, e: Edge) -> None:
        u, v = tuple(e)
        self.S.add(e)
        self.neighbors[u].add(v)
        self.neighbors[v].add(u)

    def _remove_edge_from_sample(self, e: Edge) -> None:
        if e not in self.S:
            return
        u, v = tuple(e)
        self.S.remove(e)
        self.neighbors[u].discard(v)
        self.neighbors[v].discard(u)

    def _common_neighbors(self, e: Edge) -> Set[int]:
        u, v = tuple(e)
        return self.neighbors[u] & self.neighbors[v]

    def _update_counters_with_weight(self, e: Edge, weight: float) -> None:
        if len(self.S) == 0:
            return
        u, v = tuple(e)
        common = self._common_neighbors(e)
        if not common:
            return
        inc = float(weight)
        for c in common:
            self.tau += inc
            self.tau_vertices[u] += inc
            self.tau_vertices[v] += inc
            self.tau_vertices[c] += inc

    def _sample_edge(self, e: Edge) -> Tuple[bool, Optional[Edge]]:
        if self.t <= self.M:
            return True, None
        accept_prob = self.M / float(self.t)
        if random.random() < accept_prob:
            evict = random.choice(tuple(self.S))
            return True, evict
        return False, None

    def process_insertion(self, u: int, v: int) -> None:
        e = make_edge(u, v)
        self.t += 1

        accepted, evicted = self._sample_edge(e)
        if accepted:
            if evicted is not None:
                self._update_counters_with_weight(evicted, -1.0)
                self._remove_edge_from_sample(evicted)

            self._add_edge_to_sample(e)
            self._update_counters_with_weight(e, +1.0)

    def estimate_global(self) -> float:
        return self.xi * self.tau

    def estimate_local(self, u: int) -> float:
        return self.xi * self.tau_vertices.get(u, 0.0)

class TriestImproved(TriestBase):
    def __init__(self, M: int):
        super().__init__(M)

    @property
    def eta(self) -> float:
        if self.t < 3:
            return 1.0
        numer = (self.t - 1) * (self.t - 2)
        denom = self.M * (self.M - 1)
        if denom == 0:
            return 1.0
        return max(1.0, numer / denom)

    def process_insertion(self, u: int, v: int) -> None:
        e = make_edge(u, v)
        self.t += 1
        self._update_counters_with_weight(e, self.eta)
        accepted, evicted = self._sample_edge(e)
        if accepted:
            if evicted is not None:
                self._remove_edge_from_sample(evicted)
            self._add_edge_to_sample(e)

    def estimate_global(self) -> float:
        return float(self.tau)

if __name__ == "__main__":
    start_time = time.time()
    # locate dataset relative to repository root
    ROOT_DIR = Path(__file__).resolve().parents[3]
    DATASET_PATH = ROOT_DIR / "data" / "triest" / "facebook_combined.txt"

    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset file not found: {DATASET_PATH}")

    # read edges from file (each line contains two node ids)
    edges: list[Tuple[int, int]] = []
    with open(DATASET_PATH, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            edges.append((u, v))

    print(f"Loaded {len(edges)} edges from {DATASET_PATH}")

    print("Testing TriestBase on facebook_combined stream")
    base = TriestBase(M=10000)
    for u, v in edges:
        base.process_insertion(u, v)
    print("t=", base.t, "sample size=", len(base.S), "global estimate=", base.estimate_global())
    end_time = time.time()
    print(f'\nTime needed is {end_time-start_time} seconds')
    print("\nTesting TriestImproved on facebook_combined stream")
    impr = TriestImproved(M=10000)
    for u, v in edges:
        impr.process_insertion(u, v)
    print("t=", impr.t, "sample size=", len(impr.S), "global estimate=", impr.estimate_global())
