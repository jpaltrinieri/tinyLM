from pathlib import Path

from tinylm.data import load_corpus


def test_load_corpus_drops_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("a\n\nb\n", encoding="utf-8")

    assert load_corpus(path) == ["a", "b"]


# --- Vocab -----------------------------------------------------------------

from tinylm.data import Vocab, split  # noqa: E402

WORDS = ["emma", "olivia", "ava"]


def test_vocab_size_is_distinct_chars_plus_one() -> None:
    distinct = set("".join(WORDS))
    assert len(Vocab(WORDS)) == len(distinct) + 1


def test_boundary_token_is_index_zero() -> None:
    v = Vocab(WORDS)
    assert v.decode([0]) == "."


def test_round_trip_every_word() -> None:
    v = Vocab(WORDS)
    for w in WORDS:
        assert v.decode(v.encode(w)) == w


def test_encode_is_ids_not_chars() -> None:
    v = Vocab(WORDS)
    ids = v.encode("ava")
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) == 3
    assert 0 not in ids


def test_boundary_inside_corpus_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        Vocab(["a.b"])


# --- split -----------------------------------------------------------------

NAMES = [f"name{i}" for i in range(100)]


def test_split_sizes() -> None:
    train, val, test = split(NAMES, seed=0)
    assert (len(train), len(val), len(test)) == (80, 10, 10)


def test_split_is_reproducible() -> None:
    assert split(NAMES, seed=1) == split(NAMES, seed=1)


def test_split_depends_on_seed() -> None:
    assert split(NAMES, seed=1) != split(NAMES, seed=2)


def test_split_is_disjoint_and_complete() -> None:
    train, val, test = split(NAMES, seed=3)
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert sorted(train + val + test) == sorted(NAMES)
