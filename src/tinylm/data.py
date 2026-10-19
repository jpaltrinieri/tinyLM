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
