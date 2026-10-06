from pathlib import Path

from scripts.prepare_real_repeat_control import build_repeated_rows, proportional_targets


def test_proportional_targets_preserve_long_tail_ratio() -> None:
    assert proportional_targets({"inclusion": 10, "patches": 30, "scratches": 60}, 250) == {
        "inclusion": 25,
        "patches": 75,
        "scratches": 150,
    }


def test_repeated_rows_have_exact_size_and_only_known_sources() -> None:
    paths = (
        [Path(f"inclusion_{index}.jpg") for index in range(2)]
        + [Path(f"patches_{index}.jpg") for index in range(3)]
        + [Path(f"scratches_{index}.jpg") for index in range(5)]
    )
    rows, targets = build_repeated_rows(paths, 20)
    assert len(rows) == 20
    assert targets == {"inclusion": 4, "patches": 6, "scratches": 10}
    assert set(rows) <= set(paths)
