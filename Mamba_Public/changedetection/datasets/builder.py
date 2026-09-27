import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import img_loader
from .damage_assessment import DamageAssessmentDataset
from .multimodal_damage_assessment import MultimodalDamageAssessmentDataset


def build_dataset(
    dataset_name,
    *,
    dataset_path,
    data_list,
    crop_size,
    max_iters=None,
    batch_size=1,
    split="train",
    suffix=".tif",
    data_loader=img_loader,
):
    if "xBD" in dataset_name:
        return DamageAssessmentDataset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            data_loader=data_loader,
        )
    if "BRIGHT" in dataset_name:
        return MultimodalDamageAssessmentDataset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            suffix=suffix,
            data_loader=data_loader,
        )
    raise NotImplementedError(f"Unsupported dataset: {dataset_name}")


def resolve_train_batch_size(args):
    return args.train_batch_size if "BRIGHT" in args.dataset else args.batch_size


def resolve_train_num_workers(args):
    if "BRIGHT" in args.dataset:
        return 4 if args.num_workers is None else args.num_workers
    if "xBD" in args.dataset:
        return 6
    raise NotImplementedError(f"Unsupported dataset: {args.dataset}")


def seed_data_loader_worker(worker_id):
    """Seed Python and NumPy RNGs used by image augmentations in each worker."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_train_loader(args, **kwargs):
    batch_size = resolve_train_batch_size(args)
    dataset = build_dataset(
        args.dataset,
        dataset_path=args.train_dataset_path,
        data_list=args.train_data_name_list,
        crop_size=args.crop_size,
        max_iters=args.max_iters,
        batch_size=batch_size,
        split=args.type,
    )
    loader_kwargs = dict(kwargs)
    if "generator" not in loader_kwargs and hasattr(args, "seed"):
        generator = torch.Generator()
        generator.manual_seed(int(args.seed))
        loader_kwargs["generator"] = generator
    if "worker_init_fn" not in loader_kwargs and hasattr(args, "seed"):
        loader_kwargs["worker_init_fn"] = seed_data_loader_worker

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=resolve_train_num_workers(args),
        drop_last=False,
        **loader_kwargs,
    )


def build_eval_loader(
    dataset_name,
    *,
    dataset_path,
    data_list,
    crop_size,
    split="test",
    batch_size=1,
    num_workers=4,
    suffix=".tif",
    shuffle=False,
    **kwargs,
):
    dataset = build_dataset(
        dataset_name,
        dataset_path=dataset_path,
        data_list=data_list,
        crop_size=crop_size,
        batch_size=batch_size,
        split=split,
        suffix=suffix,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        **kwargs,
    )


def make_data_loader(args, **kwargs):
    return build_train_loader(args, **kwargs)
