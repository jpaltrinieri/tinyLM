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
    """Character vocabulary with one special boundary token at index 0.

    The boundary token marks the start and end of a string. It must not
    appear inside any string of the corpus.
    """

    def __init__(self, words: list[str], boundary: str = ".") -> None:
        """Build the vocabulary from the distinct characters in `words`.

        Index 0 is `boundary`; the remaining indices are the distinct
        characters of `words` in sorted order.

        Raises:
            ValueError: if `boundary` occurs inside any word.
        """
        if any(boundary in w for w in words):
            raise ValueError(f"boundary token {boundary!r} occurs inside the corpus")
        self.boundary = boundary
        self.itos: list[str] = [boundary] + sorted(set("".join(words)))
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self) -> int:
        """Number of tokens, including the boundary token."""
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        """Map a string to a list of token ids, without boundary tokens."""
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        """Map a list of token ids back to a string."""
        return "".join(self.itos[i] for i in ids)


def split(
    words: list[str], seed: int, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle `words` with `seed` and cut into train, validation and test.

    The same seed always gives the same split. Use `random.Random(seed)`,
    not the module-level functions, so global state is untouched.
    """
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test
