"""
Reorganize cropped fish images into a train/val classifier dataset.

Usage:
    python organize_classifier_dataset.py SRC_DIR TARGET_DIR [--train-fraction F]

Input layout (SRC_DIR):
    <SRC_DIR>/
        <folder_id>_<rl>/        # e.g. 240101_01_some_label_L
            correct/             # correctly labelled crop images
            incorrect/           # incorrectly labelled crop images
        ...

Output layout (TARGET_DIR):
    <TARGET_DIR>/
        train/
            correct/
            incorrect/
        val/
            correct/
            incorrect/

- Each source folder is assigned entirely to either train or val (no folder is split across splits). 
- Folders are stratified by their correct/total ratio before assignment so that both splits cover the 
  full easy-to-hard range.
- Train folders are then filled greedily (largest first) until train_fraction * total_images is reached 
  (default 80 %).
- Output filenames are prefixed with <folder_id>_<L|R>_ to avoid collisions when images from different 
  folders share the same base name.
"""

# Standard Library imports
import argparse
import re
import shutil
from collections import namedtuple
from pathlib import Path

# External imports
from tqdm import tqdm


TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT)
CLASS_NAMES = ("correct", "incorrect")
FOLDER_PATTERN = re.compile(r".*(\d{6}_\d{2}).*_(L|R)")

FolderStats = namedtuple(
    "FolderStats", ["folder", "folder_id", "rl", "correct", "incorrect", "total"]
)


def existing_dir(path: str) -> Path:
    """argparse type that validates the path exists."""
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{path} doesn't exist")
    return p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize cropped fish images into train/val splits.",
        usage="%(prog)s src_dir target_dir [--train-fraction F]",
    )
    parser.add_argument("src_dir", type=existing_dir)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Target fraction of total images for the train split (default: 0.8)",
    )
    return parser.parse_args()


def count_folder_stats(src_dir: Path) -> list[FolderStats]:
    """Scan source folders and count images per class."""
    stats = []
    for folder in src_dir.iterdir():
        m = FOLDER_PATTERN.match(folder.name)
        if not m:
            print(f"Skipping folder with unexpected name: {folder.name}")
            continue
        folder_id, rl = m.groups()
        correct_n = (
            len(list((folder / "correct").iterdir()))
            if (folder / "correct").exists()
            else 0
        )
        incorrect_n = (
            len(list((folder / "incorrect").iterdir()))
            if (folder / "incorrect").exists()
            else 0
        )
        stats.append(
            FolderStats(
                folder, folder_id, rl, correct_n, incorrect_n, correct_n + incorrect_n
            )
        )
    return stats


def select_train_folders(stats: list[FolderStats], train_fraction: float) -> set[Path]:
    """Select train folders so both splits have a similar correct/incorrect ratio.

    Folders are sorted by correct/total ratio and interleaved into train/val
    candidate pools (alternating indices), so each pool spans the full
    easy-to-hard range. Train folders are then filled greedily (largest-first)
    until train_fraction * total_images is reached.
    """
    sorted_stats = sorted(
        stats, key=lambda s: s.correct / s.total if s.total > 0 else 0
    )

    # Interleave: alternate folders between train/val candidate pools
    train_candidates = sorted_stats[0::2]  # indices 0, 2, 4, ...
    val_candidates = sorted_stats[1::2]  # indices 1, 3, 5, ...

    grand_total = sum(s.total for s in stats)
    train_budget = int(train_fraction * grand_total)

    # Greedy fill train candidates (largest first) until budget is met
    train_folders: set[Path] = set()
    cumulative = 0
    for s in sorted(train_candidates, key=lambda s: s.total, reverse=True):
        if cumulative >= train_budget:
            break
        train_folders.add(s.folder)
        cumulative += s.total

    # If budget not met, pull from val candidates (largest first)
    for s in sorted(val_candidates, key=lambda s: s.total, reverse=True):
        if cumulative >= train_budget:
            break
        train_folders.add(s.folder)
        cumulative += s.total

    return train_folders


def print_split_summary(stats: list[FolderStats], train_folders: set[Path]) -> None:
    """Print per-folder table and split-level stats."""
    print(
        f"\n{'Folder':<50} {'Correct':>8} {'Incorrect':>10} {'Total':>7} {'Split':>6}"
    )
    print("-" * 85)
    for s in sorted(stats, key=lambda s: s.correct / s.total if s.total > 0 else 0):
        split = TRAIN_SPLIT if s.folder in train_folders else VAL_SPLIT
        print(
            f"{s.folder.name:<50} {s.correct:>8} {s.incorrect:>10} {s.total:>7} {split:>6}"
        )

    train_stats = [s for s in stats if s.folder in train_folders]
    val_stats = [s for s in stats if s.folder not in train_folders]

    def summarise(subset, label):
        n_folders = len(subset)
        n_correct = sum(s.correct for s in subset)
        n_incorrect = sum(s.incorrect for s in subset)
        n_total = n_correct + n_incorrect
        ratio = n_correct / n_total if n_total > 0 else float("nan")
        grand = sum(s.total for s in stats)
        frac = n_total / grand * 100 if grand > 0 else 0
        print(
            f"  {label}: {n_folders} folders | {n_total} images ({frac:.1f}%) | correct {ratio:.2%}"
        )

    grand_total = sum(s.total for s in stats)
    print(f"\nSplit summary (total {len(stats)} folders, {grand_total} images):")
    summarise(train_stats, "train")
    summarise(val_stats, "val  ")
    print()


def main():
    """Reorganize cropped fish images from source folders into train/val splits."""

    args = parse_args()
    src_dir = args.src_dir
    target_dir = args.target_dir
    train_fraction = args.train_fraction

    # Create target dirs
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # Count images per folder
    print("Counting images per folder...")
    stats = count_folder_stats(src_dir)
    if not stats:
        print("No valid folders found.")
        return

    # Determine train/val assignment
    train_folders = select_train_folders(stats, train_fraction)

    print_split_summary(stats, train_folders)

    print("Copying files...")
    for s in tqdm(stats):
        split = TRAIN_SPLIT if s.folder in train_folders else VAL_SPLIT
        for class_name in CLASS_NAMES:
            class_dir = s.folder / class_name
            if not class_dir.exists():
                continue
            for f in class_dir.iterdir():
                new_name = f"{s.folder_id}_{s.rl}_{f.name}"
                new_filepath = target_dir / split / class_name / new_name
                if new_filepath.exists():
                    continue
                shutil.copy(f, new_filepath)


if __name__ == "__main__":
    main()
