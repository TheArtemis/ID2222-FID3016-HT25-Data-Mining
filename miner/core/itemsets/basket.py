from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel


class Basket(BaseModel):
    itemset: set[int]  # Hashed set of items

    def __contains__(self, item: int) -> bool:
        return item in self.itemset

    @staticmethod
    def load(file_path: Path) -> list[Basket]:
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        with open(file_path) as f:
            return [Basket(itemset=set(map(int, line.split()))) for line in f]
