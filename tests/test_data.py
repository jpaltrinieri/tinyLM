from pathlib import Path

from tinylm.data import load_corpus


def test_load_corpus_drops_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("a\n\nb\n", encoding="utf-8")

    assert load_corpus(path) == ["a", "b"]
