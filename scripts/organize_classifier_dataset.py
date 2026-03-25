"""
Reorganize cropped fish images into a train/val/test classifier dataset.

Usage:
    python organize_classifier_dataset.py src_dir target_dir --test-folders FILE [--train-fraction F]

Arguments:
    src_dir          Source directory containing labelled crop folders.
    target_dir       Destination directory where train/val/test splits will be written.
    --test-folders   Text file listing folder names (one per line) to reserve for the test split.
    --train-fraction Target fraction of total images for the train split (default: 0.8, must be in (0, 1)).

Input layout (src_dir):
    <src_dir>/
        <folder_id>_<rl>/        # e.g. 240101_01_some_label_L
            correct/             # correctly labelled crop images
            incorrect/           # incorrectly labelled crop images
        ...

Output layout (target_dir):
    <target_dir>/
        train/
            correct/
            incorrect/
        val/
            correct/
            incorrect/
        test/
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
from collections import defaultdict, namedtuple
from pathlib import Path

# External imports
from tqdm import tqdm

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
CLASS_NAMES = ("correct", "incorrect")
FOLDER_PATTERN = re.compile(r".*(\d{6}_\d{2}).*_(L|R)")

FolderStats = namedtuple(
    "FolderStats", ["folder_path", "folder_id", "rl", "correct", "incorrect", "total"]
)

StereoPair = namedtuple("StereoPair", ["folder_id", "members", "correct", "total"])


def existing_dir(path: str) -> Path:
    """argparse type that validates the path exists."""
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{path} doesn't exist")
    return p


def existing_file(path: str) -> Path:
    """argparse type that validates the path is an existing file."""
    p = Path(path)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"{path} doesn't exist or is not a file")
    return p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize cropped fish images into train/val/test splits.",
        usage="%(prog)s src_dir target_dir --test-folders FILE [--train-fraction F]",
    )
    parser.add_argument("src_dir", type=existing_dir)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument(
        "--test-folders",
        type=existing_file,
        required=True,
        help="Text file listing folder names (one per line) to reserve for the test split",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Target fraction of total images for the train split (default: 0.8)",
    )
    return parser.parse_args()


def _count_images(folder: Path, class_name: str) -> int:
    d = folder / class_name
    return len(list(d.glob("*.png"))) if d.exists() else 0


def _check_test_pair_integrity(stats: list[FolderStats], src_dir: Path) -> None:
    """Raise if one side of a stereo pair is in the test set but its counterpart is not.

    Splitting a stereo pair across splits would leak correlated images between sets.
    """
    test_by_id: dict[str, list[FolderStats]] = defaultdict(list)
    for s in stats:
        test_by_id[s.folder_id].append(s)

    for folder_id, members in test_by_id.items():
        if len(members) >= 2:
            continue
        present_rl = members[0].rl
        opposite_rl = "R" if present_rl == "L" else "L"
        for candidate in src_dir.iterdir():
            if not candidate.is_dir():
                continue
            m = FOLDER_PATTERN.match(candidate.name)
            if m and m.group(1) == folder_id and m.group(2) == opposite_rl:
                raise ValueError(
                    f"Stereo pair is split across splits: test file includes "
                    f"{members[0].folder_path.name!r} but not its {opposite_rl} "
                    f"counterpart {candidate.name!r}. Add the counterpart to the "
                    f"test file or remove the listed folder."
                )


def read_test_folder_stats(test_folders_file: Path, src_dir: Path) -> list[FolderStats]:
    """Build FolderStats for folders listed in a text file (one folder name per line)."""
    stats = []
    for line in test_folders_file.read_text().splitlines():
        name = line.strip()
        if not name:
            continue
        folder = src_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Test folder listed in {test_folders_file} does not exist: {folder}"
            )
        m = FOLDER_PATTERN.match(name)
        if not m:
            raise ValueError(
                f"Test folder name doesn't match expected pattern ({FOLDER_PATTERN.pattern}): {name}"
            )
        folder_id, rl = m.groups()
        correct_n = _count_images(folder, CLASS_NAMES[0])
        incorrect_n = _count_images(folder, CLASS_NAMES[1])
        stats.append(
            FolderStats(
                folder, folder_id, rl, correct_n, incorrect_n, correct_n + incorrect_n
            )
        )
    _check_test_pair_integrity(stats, src_dir)
    return stats


def count_folder_stats(
    src_dir: Path, exclude: set[str] | None = None
) -> list[FolderStats]:
    """Scan source folders and count images per class."""
    stats = []
    for folder in src_dir.iterdir():
        if not folder.is_dir():
            continue
        if exclude and folder.name in exclude:
            continue

        m = FOLDER_PATTERN.match(folder.name)
        if not m:
            print(f"Skipping folder with unexpected name: {folder.name}")
            continue

        folder_id, rl = m.groups()
        correct_n = _count_images(folder, CLASS_NAMES[0])
        incorrect_n = _count_images(folder, CLASS_NAMES[1])
        stats.append(
            FolderStats(
                folder, folder_id, rl, correct_n, incorrect_n, correct_n + incorrect_n
            )
        )
    return stats


def _build_stereo_pairs(stats: list[FolderStats]) -> list[StereoPair]:
    """Group FolderStats by folder_id so each stereo pair (L+R) is one unit."""
    groups: dict[str, list[FolderStats]] = defaultdict(list)
    for s in stats:
        groups[s.folder_id].append(s)
    return [
        StereoPair(
            folder_id=fid,
            members=members,
            correct=sum(s.correct for s in members),
            total=sum(s.total for s in members),
        )
        for fid, members in groups.items()
    ]


def _greedy_fill(
    candidates: list[StereoPair], budget: int, cumulative: int
) -> tuple[set[Path], int]:
    """Add pairs (largest-first) until the image budget is reached."""
    train_folders: set[Path] = set()
    for pair in sorted(candidates, key=lambda p: p.total, reverse=True):
        if cumulative >= budget:
            break
        for s in pair.members:
            train_folders.add(s.folder_path)
        cumulative += pair.total
    return train_folders, cumulative


def select_train_folders(stats: list[FolderStats], train_fraction: float) -> set[Path]:
    """Select train folders so both splits have a similar correct/incorrect ratio.

    Stereo pairs (folders sharing the same folder_id, differing only by L/R)
    are kept together: both go to train or both go to val.

    Pairs are sorted by their aggregate correct/total ratio and interleaved
    into train/val candidate pools (even/odd indices), so each pool spans the
    full easy-to-hard range. Train pairs are then filled greedily
    (largest-first by total images) until train_fraction * total_images is
    reached. Val candidates are used as overflow if needed.
    """
    pairs = _build_stereo_pairs(stats)

    # Sort by difficulty (correct ratio), then interleave into two pools
    pairs_by_difficulty = sorted(
        pairs, key=lambda p: p.correct / p.total if p.total > 0 else 0
    )
    train_candidates = pairs_by_difficulty[0::2]  # even indices
    val_candidates = pairs_by_difficulty[1::2]  # odd indices

    train_budget = int(train_fraction * sum(s.total for s in stats))
    train_folders, cumulative = _greedy_fill(train_candidates, train_budget, 0)

    # Draw from val candidates if train budget is still not met
    if cumulative < train_budget:
        overflow, _ = _greedy_fill(val_candidates, train_budget, cumulative)
        train_folders |= overflow

    return train_folders


def print_split_summary(
    train_stats: list[FolderStats],
    val_stats: list[FolderStats],
    test_stats: list[FolderStats],
) -> None:
    """Print per-folder table and split-level stats."""
    all_stats = train_stats + val_stats + test_stats
    grand_total = sum(s.total for s in all_stats)
    split_map = (
        {s.folder_path: TRAIN_SPLIT for s in train_stats}
        | {s.folder_path: VAL_SPLIT for s in val_stats}
        | {s.folder_path: TEST_SPLIT for s in test_stats}
    )

    def summarise(subset: list[FolderStats], label: str) -> None:
        n_total = sum(s.total for s in subset)
        ratio = (
            sum(s.correct for s in subset) / n_total if n_total > 0 else float("nan")
        )
        frac = n_total / grand_total * 100 if grand_total > 0 else 0
        print(
            f"  {label}: {len(subset)} folders | {n_total} images ({frac:.1f}%) | correct {ratio:.2%}"
        )

    print(
        f"\n{'Folder':<50} {'Correct':>8} {'Incorrect':>10} {'Total':>7} {'Split':>6}"
    )
    print("-" * 85)
    for s in sorted(all_stats, key=lambda s: s.folder_path.name):
        print(
            f"{s.folder_path.name:<50} {s.correct:>8} {s.incorrect:>10} {s.total:>7} {split_map[s.folder_path]:>6}"
        )

    print(f"\nSplit summary (total {len(all_stats)} folders, {grand_total} images):")
    summarise(train_stats, "train")
    summarise(val_stats, "val  ")
    summarise(test_stats, "test ")
    print()


def copy_images_to_split(
    stats: list[FolderStats], target_dir: Path, split: str
) -> None:
    """Copy all images from a list of FolderStats into target_dir/split/."""
    for s in tqdm(stats, desc=f"Copying {split}"):
        for class_name in CLASS_NAMES:
            class_dir = s.folder_path / class_name
            if not class_dir.exists():
                continue
            for f in class_dir.glob("*.png"):
                new_name = f"{s.folder_id}_{s.rl}_{f.name}"
                new_filepath = target_dir / split / class_name / new_name
                if new_filepath.exists():
                    continue
                shutil.copy(f, new_filepath)


def main():
    """Reorganize cropped fish images from source folders into train/val/test splits."""

    args = parse_args()
    src_dir = args.src_dir
    target_dir = args.target_dir
    train_fraction = args.train_fraction

    if not (0 < train_fraction < 1):
        raise ValueError(
            f"--train-fraction must be between 0 and 1 (exclusive), got {train_fraction}"
        )

    print("Counting images in test split")
    test_folder_stats = read_test_folder_stats(args.test_folders, src_dir)
    test_folder_names = {s.folder_path.name for s in test_folder_stats}

    print("Counting images in train/val splits")
    train_val_stats = count_folder_stats(src_dir, exclude=test_folder_names)
    train_folders = select_train_folders(train_val_stats, train_fraction)
    train_folder_stats = [s for s in train_val_stats if s.folder_path in train_folders]
    val_folder_stats = [
        s for s in train_val_stats if s.folder_path not in train_folders
    ]

    if not train_folder_stats:
        raise ValueError("No valid train folders found.")
    if not val_folder_stats:
        raise ValueError("No valid val folders found.")
    if not test_folder_stats:
        raise ValueError("No valid test folders found.")

    # Create target split dirs
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    print_split_summary(train_folder_stats, val_folder_stats, test_folder_stats)

    answer = input(f"Copy files to {target_dir}? [y/n] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    copy_images_to_split(train_folder_stats, target_dir, TRAIN_SPLIT)
    copy_images_to_split(val_folder_stats, target_dir, VAL_SPLIT)
    copy_images_to_split(test_folder_stats, target_dir, TEST_SPLIT)


if __name__ == "__main__":
    main()
