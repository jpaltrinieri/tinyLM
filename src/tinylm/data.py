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
    """
    Class for Vocabulary, a list of strings and its encoding / decoding, 0 is the boundary "."
    """

    def __init__(self, words: list[str], boundary: str = ".") -> None:
        if any(boundary in w for w in words):
            raise ValueError("Boundary cannot be in the vocabulary")
        self.itos = sorted(set(".".join(words)))
        self.stoi = {j: i for i, j in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[i] for i in s]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.itos[d] for d in ids])
