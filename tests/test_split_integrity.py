from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = PROJECT / "data" / "splits" / "v1"


def _paths(name: str) -> set[str]:
    return {
        line.strip()
        for line in (SPLIT_ROOT / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_splits_are_disjoint_and_expected_size() -> None:
    train, val, test = _paths("train_fewshot"), _paths("val"), _paths("test")
    assert len(train) == 100
    assert len(val) == 150
    assert len(test) == 150
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert all(Path(path).exists() for path in train | val | test)

