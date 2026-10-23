import random
from pathlib import Path


def load_corpus(path: Path) -> list[str]:
    """
    Load a text corpus from a file.

    Args:
        path (Path): The path to the text file containing the corpus.

    Returns:
        List[str]: A list of strings representing the sentences in the corpus.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    result = [line.strip() for line in lines]
    return [item for item in result if item != ""]


class Vocab:
    """Character vocabulary; index 0 is the boundary token."""

    def __init__(self, words: list[str], boundary: str = ".") -> None:
        if any(boundary in w for w in words):
            raise ValueError("Boundary cannot be in the vocabulary")
        self.itos = [boundary] + sorted(set("".join(words)))
        self.stoi = {j: i for i, j in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[i] for i in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[d] for d in ids)


def split(words: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    """Shuffle a copy of `words` with a private generator seeded by `seed`; cut 80/10/10.

    Returns (train, validation, test). The input list is not modified.
    """
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]
