import json
import random
from pathlib import Path

TRAIN_FILE = "train_id.json"
VAL_FILE = "val_id.json"
TEST_FILE = "test_id.json"

NEW_TRAIN_FILE = "train_id.json"
NEW_VAL_FILE = "val_id.json"
NEW_TEST_FILE = "test_id.json"

SEED = 42


def load_json_list(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list.")
    return data


def main():
    train_samples = load_json_list(Path(TRAIN_FILE))
    val_samples = load_json_list(Path(VAL_FILE))
    test_samples = load_json_list(Path(TEST_FILE))

    merged = train_samples + val_samples + test_samples

    if len(set(merged)) != len(merged):
        raise ValueError("Duplicate samples found across train/val/test before merging.")

    rng = random.Random(SEED)
    rng.shuffle(merged)

    n = len(merged)
    n_train = int(round(n * 0.60))
    n_val = int(round(n * 0.30))

    if n_train + n_val > n:
        n_val = n - n_train

    new_train = merged[:n_train]
    new_val = merged[n_train:n_train + n_val]
    new_test = merged[n_train + n_val:]

    assert len(new_train) + len(new_val) + len(new_test) == n

    with open(NEW_TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(new_train, f, indent=2, ensure_ascii=False)

    with open(NEW_VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(new_val, f, indent=2, ensure_ascii=False)

    with open(NEW_TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(new_test, f, indent=2, ensure_ascii=False)

    print("[Done] Re-split completed with seed =", SEED)
    print(f"Total merged samples : {len(merged)}")
    print(f"Train total          : {len(new_train)}")
    print(f"Val total            : {len(new_val)}")
    print(f"Test total           : {len(new_test)}")


if __name__ == "__main__":
    main()