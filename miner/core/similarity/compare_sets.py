class CompareSets:
    def __init__(self):
        pass

    def jaccard_similarity(self, set1: list[int], set2: list[int]) -> float:
        union = set(set1) | set(set2)
        intersection = set(set1) & set(set2)
        return len(intersection) / len(union)
