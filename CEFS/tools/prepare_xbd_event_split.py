#!/usr/bin/env python3
"""Create a DisasterAdaptiveNet-style event split for ChangeMamba xBD.

The split follows the public DisasterAdaptiveNet event protocol:
13 source events are split into train/validation with a stratified 90/10 split,
and 6 held-out events form the test set. The script builds ChangeMamba-readable
folders using links or copies, because ChangeMamba expects different filename
forms for training and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TRAIN_EVENTS = (
    "lower-puna-volcano",
    "palu-tsunami",
    "mexico-earthquake",
    "socal-fire",
    "woolsey-fire",
    "portugal-wildfire",
    "pinery-bushfire",
    "midwest-flooding",
    "moore-tornado",
    "joplin-tornado",
    "hurricane-harvey",
    "hurricane-michael",
    "hurricane-florence",
)

TEST_EVENTS = (
    "nepal-flooding",
    "guatemala-volcano",
    "sunda-tsunami",
    "santa-rosa-wildfire",
    "hurricane-matthew",
    "tuscaloosa-tornado",
)


@dataclass(frozen=True)
class Sample:
    source_list: str
    source_subset: str
    raw_name: str
    event: str
    patch_id: str

    @property
    def base_name(self) -> str:
        return f"{self.event}_{self.patch_id}"

    @property
    def train_item_name(self) -> str:
        return f"{self.base_name}_0_0"

    @property
    def eval_item_name(self) -> str:
        return self.base_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: DATASET_ROOT/event_splits_v1",
    )
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize split folders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate items in {path}")
    return names


def parse_sample(source_list: str, source_subset: str, item: str) -> Sample:
    name = Path(item).name
    if name.endswith(".png"):
        name = name[:-4]
    for marker in ("_pre_disaster", "_post_disaster"):
        if marker in name:
            name = name.split(marker, 1)[0]
    match = re.match(r"^(?P<event>.+?)_(?P<patch>[0-9]{6,})(?:_0_0)?$", name)
    if not match:
        raise ValueError(f"Cannot parse xBD sample name: {item!r}")
    return Sample(
        source_list=source_list,
        source_subset=source_subset,
        raw_name=item,
        event=match.group("event").lower(),
        patch_id=match.group("patch"),
    )


def collect_samples(root: Path) -> list[Sample]:
    specs = (
        ("train_set.txt", "train"),
        ("test_set.txt", "test"),
        ("hold_set.txt", "hold"),
    )
    samples: list[Sample] = []
    seen: set[str] = set()
    for list_name, subset in specs:
        for item in read_list(root / list_name):
            sample = parse_sample(list_name, subset, item)
            if sample.base_name in seen:
                raise ValueError(f"Duplicate base sample across lists: {sample.base_name}")
            seen.add(sample.base_name)
            samples.append(sample)
    return samples


def stratified_train_val(samples: Sequence[Sample], val_ratio: float, seed: int):
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
    except Exception:
        return fallback_stratified_train_val(samples, val_ratio, seed)

    indices = np.arange(len(samples))
    labels = [sample.event for sample in samples]
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return [samples[int(i)] for i in train_indices], [samples[int(i)] for i in val_indices]


def fallback_stratified_train_val(samples: Sequence[Sample], val_ratio: float, seed: int):
    """Dependency-free stratified split used when scikit-learn is unavailable."""
    rng = random.Random(seed)
    by_event: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_event[sample.event].append(sample)

    target_val = math.ceil(len(samples) * val_ratio)
    exact = {event: len(items) * val_ratio for event, items in by_event.items()}
    val_counts = {event: int(value) for event, value in exact.items()}
    remaining = target_val - sum(val_counts.values())
    remainders = sorted(
        ((exact[event] - val_counts[event], rng.random(), event) for event in by_event),
        reverse=True,
    )
    for _, _, event in remainders[:remaining]:
        val_counts[event] += 1

    train: list[Sample] = []
    val: list[Sample] = []
    for event in sorted(by_event):
        items = list(by_event[event])
        rng.shuffle(items)
        val_count = val_counts[event]
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def source_paths(root: Path, sample: Sample) -> tuple[Path, Path, Path, Path]:
    if sample.source_subset == "train":
        pre = f"{sample.base_name}_pre_disaster_0_0.png"
        post = f"{sample.base_name}_post_disaster_0_0.png"
    else:
        pre = f"{sample.base_name}_pre_disaster.png"
        post = f"{sample.base_name}_post_disaster.png"
    subset_root = root / sample.source_subset
    return (
        subset_root / "images" / pre,
        subset_root / "images" / post,
        subset_root / "masks" / pre,
        subset_root / "masks" / post,
    )


def dest_names(sample: Sample, split: str) -> tuple[str, str]:
    if split == "train":
        return (
            f"{sample.base_name}_pre_disaster_0_0.png",
            f"{sample.base_name}_post_disaster_0_0.png",
        )
    return (
        f"{sample.base_name}_pre_disaster.png",
        f"{sample.base_name}_post_disaster.png",
    )


def materialize(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; re-run with --overwrite")
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        rel_src = os.path.relpath(src, start=dst.parent)
        os.symlink(rel_src, dst)


def write_text(path: Path, lines: Iterable[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; re-run with --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def build_split(root: Path, output: Path, split: str, samples: Sequence[Sample], mode: str, overwrite: bool):
    image_dir = output / "data" / split / "images"
    mask_dir = output / "data" / split / "masks"
    list_names = []
    for sample in samples:
        src_pre_img, src_post_img, src_pre_mask, src_post_mask = source_paths(root, sample)
        dst_pre, dst_post = dest_names(sample, split)
        materialize(src_pre_img, image_dir / dst_pre, mode, overwrite)
        materialize(src_post_img, image_dir / dst_post, mode, overwrite)
        materialize(src_pre_mask, mask_dir / dst_pre, mode, overwrite)
        materialize(src_post_mask, mask_dir / dst_post, mode, overwrite)
        list_names.append(sample.train_item_name if split == "train" else sample.eval_item_name)
    write_text(output / f"{split}_set.txt", list_names, overwrite)


def counter_dict(samples: Sequence[Sample], attr: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(sample, attr) for sample in samples).items()))


def write_audit(output: Path, train: Sequence[Sample], val: Sequence[Sample], test: Sequence[Sample], overwrite: bool):
    rows = []
    for split, samples in (("train", train), ("val", val), ("test", test)):
        for sample in samples:
            rows.append(
                {
                    "split": split,
                    "source_list": sample.source_list,
                    "source_subset": sample.source_subset,
                    "item": sample.train_item_name if split == "train" else sample.eval_item_name,
                    "event": sample.event,
                    "patch_id": sample.patch_id,
                }
            )
    csv_path = output / "event_split_manifest.csv"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"{csv_path} exists; re-run with --overwrite")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "source_list", "source_subset", "item", "event", "patch_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    train_events = set(sample.event for sample in train)
    val_events = set(sample.event for sample in val)
    test_events = set(sample.event for sample in test)
    summary = {
        "protocol": "DisasterAdaptiveNet-style xBD event split reconstructed for ChangeMamba",
        "train_events_official": list(TRAIN_EVENTS),
        "test_events_official": list(TEST_EVENTS),
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "events_by_split": {
            "train": sorted(train_events),
            "val": sorted(val_events),
            "test": sorted(test_events),
        },
        "event_counts": {
            "train": counter_dict(train, "event"),
            "val": counter_dict(val, "event"),
            "test": counter_dict(test, "event"),
        },
        "source_subset_counts": {
            "train": counter_dict(train, "source_subset"),
            "val": counter_dict(val, "source_subset"),
            "test": counter_dict(test, "source_subset"),
        },
        "audit": {
            "train_val_share_source_events_by_design": bool(train_events & val_events),
            "test_disjoint_from_train": not bool(test_events & train_events),
            "test_disjoint_from_val": not bool(test_events & val_events),
            "all_samples_total": len(train) + len(val) + len(test),
        },
    }
    json_path = output / "event_split_summary.json"
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"{json_path} exists; re-run with --overwrite")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit_md = output / "event_split_audit.md"
    lines = [
        "# DisasterAdaptiveNet-style xBD Event Split Audit",
        "",
        "This split uses the public DisasterAdaptiveNet event protocol: source events are",
        "split into train/validation, while test events are held out by event name.",
        "",
        "| split | samples | unique events | from train | from test | from hold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, samples in (("train", train), ("val", val), ("test", test)):
        source_counts = Counter(sample.source_subset for sample in samples)
        lines.append(
            f"| {split} | {len(samples)} | {len(set(sample.event for sample in samples))} | "
            f"{source_counts.get('train', 0)} | {source_counts.get('test', 0)} | "
            f"{source_counts.get('hold', 0)} |"
        )
    lines.extend(
        [
            "",
            f"- Test disjoint from train events: `{summary['audit']['test_disjoint_from_train']}`",
            f"- Test disjoint from val events: `{summary['audit']['test_disjoint_from_val']}`",
            "- Train and val share source events by DisasterAdaptiveNet design.",
            "",
            "## Test Events",
            "",
            ", ".join(sorted(test_events)),
        ]
    )
    write_text(audit_md, lines, overwrite)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = (args.output_dir or root / "event_splits_v1").expanduser().resolve()

    samples = collect_samples(root)
    trainval = [sample for sample in samples if sample.event in TRAIN_EVENTS]
    test = [sample for sample in samples if sample.event in TEST_EVENTS]
    unknown = [sample for sample in samples if sample.event not in TRAIN_EVENTS and sample.event not in TEST_EVENTS]
    if unknown:
        raise ValueError(f"Unknown event(s): {sorted(set(sample.event for sample in unknown))}")

    train, val = stratified_train_val(trainval, args.val_ratio, args.seed)
    build_split(root, output, "train", train, args.link_mode, args.overwrite)
    build_split(root, output, "val", val, args.link_mode, args.overwrite)
    build_split(root, output, "test", test, args.link_mode, args.overwrite)
    write_audit(output, train, val, test, args.overwrite)

    print(json.dumps({
        "output_dir": str(output),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
