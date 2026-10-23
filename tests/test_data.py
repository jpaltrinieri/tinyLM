from pathlib import Path

import pytest

from tinylm.data import Vocab, load_corpus, split

WORDS = ["emma", "olivia", "ava"]

NAMES = [f"name{i}" for i in range(100)]


def test_load_corpus_drops_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("a\n\nb\n", encoding="utf-8")

    assert load_corpus(path) == ["a", "b"]


def test_size_is_distinct_chars_plus_one() -> None:
    size_v = len(Vocab(WORDS))
    exp_size = len(set("".join(WORDS))) + 1
    assert size_v == exp_size


def test_boundary_is_index_0() -> None:
    assert Vocab(WORDS).decode([0]) == "."


def test_round_trip() -> None:
    v = Vocab(WORDS)
    for w in WORDS:
        assert v.decode(v.encode(w)) == w


def test_encode_give_ids() -> None:
    v = Vocab(WORDS)
    ids = v.encode("ava")
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)
    assert 0 not in ids


def test_boundary_in_corpus_rejected() -> None:
    with pytest.raises(ValueError):
        Vocab(["a.b"])


def test_custom_boundary_is_index_0() -> None:
    assert Vocab(["ab"], boundary="~").decode([0]) == "~"


def test_split_sizes() -> None:
    train, val, test = split(NAMES, seed=1)
    assert (len(train), len(val), len(test)) == (80, 10, 10)


def test_split_same_seed_same_result() -> None:
    assert split(NAMES, seed=1) == split(NAMES, seed=1)


def test_split_different_seed_different_result() -> None:
    train1, _, _ = split(NAMES, seed=1)
    train2, _, _ = split(NAMES, seed=2)
    assert train1 != train2


def test_split_is_disjoint_and_complete() -> None:
    train, val, test = split(NAMES, seed=1)
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert sorted(train + val + test) == sorted(NAMES)
