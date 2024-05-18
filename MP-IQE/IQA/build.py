
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

from .iqa_dataset import *
from .samplers import SubsetRandomSampler, IQAPatchDistributedSampler, RandomIdentitySampler
from .iqa_dataset_clip import *

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_transform(is_train, config):
    if config.DATA.DATASET == "koniq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "livec":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "live":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "tid2013":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "csiq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "kadid":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "spaq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    # transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "livefb":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    return transform


def build_clip_transform1(config, dataset=None):
    if dataset == "livefb":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 512)),
                transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    elif dataset == "koniq":
        print("koniq transform")
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((512, 384)),
                transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    return transform


def build_clip_transform2(config, dataset=None):
    if dataset == "livefb":
        transform = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 512)),
                transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    elif dataset == "koniq":
        print("koniq transform")
        transform = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.Resize((512, 384)),
                transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    return transform


def build_IQA_dataset(config):
    print(config.DATA.DATASET)
    if config.DATA.DATASET == "koniq":
        train_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "livec":
        train_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "live":
        train_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "tid2013":
        train_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "csiq":
        train_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "kadid":
        train_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "spaq":
        train_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "livefb":
        train_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")

    return train_dataset, test_dataset


def build_CLIP_IQA_dataset(config, num):
    print(config.DATA.DATASET)
    if config.DATA.DATASET == "koniq":
        if num == 1:
            train_dataset = KONIQDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config, "koniq"),
            )
        else:
            train_dataset = KONIQDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config, "koniq"),
            )
    elif config.DATA.DATASET == "livec":
        if num == 1:
            train_dataset = LIVECDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config),
            )
        else:
            train_dataset = LIVECDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config),
            )
    elif config.DATA.DATASET == "live":
        if num == 1:
            train_dataset = LIVEDataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config),
            )
        else:
            train_dataset = LIVEDataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config),
            )
    elif config.DATA.DATASET == "tid2013":
        if num == 1:
            train_dataset = TID2013Dataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config),
            )
        else:
            train_dataset = TID2013Dataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config),
            )
    elif config.DATA.DATASET == "csiq":
        if num == 1:
            train_dataset = CSIQDataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config),
            )
        else:
            train_dataset = CSIQDataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config),
            )
    elif config.DATA.DATASET == "kadid":
        if num == 1:
            train_dataset = KADIDDataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config),
            )
        else:
            train_dataset = KADIDDataset_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config),
            )
    elif config.DATA.DATASET == "spaq":
        if num == 1:
            train_dataset = SPAQDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config),
            )
        else:
            train_dataset = SPAQDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config),
            )
    elif config.DATA.DATASET == "livefb":
        if num == 1:
            train_dataset = FBLIVEFolder_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config, "livefb"),
            )
        else:
            train_dataset = FBLIVEFolder_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config, "livefb"),
            )
    elif config.DATA.DATASET == "bid":
        if num == 1:
            train_dataset = BIDDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform1(config, "livefb"),
            )
        else:
            train_dataset = BIDDATASET_clip(
                config.DATA.DATA_PATH,
                config.SET.TRAIN_INDEX,
                config.DATA.PATCH_NUM,
                transform=build_clip_transform2(config, "livefb"),
            )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")
    return train_dataset


def IQA_build_loader(config):
    config.defrost()
    dataset_train, dataset_val = build_IQA_dataset(config=config)
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = IQAPatchDistributedSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataloader(config, dataset_name, dataset_path, batch_size, test_index):
    config.defrost()
    if dataset_name == "koniq":
        test_dataset = KONIQDATASET_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config)
        )
    elif dataset_name == "livec":
        test_dataset = LIVECDATASET_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config)
        )
    elif dataset_name == "live":
        test_dataset = LIVEDataset_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config)
        )
    elif dataset_name == "tid2013":
        test_dataset = TID2013Dataset_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config)
        )
    elif dataset_name == "csiq":
        test_dataset = CSIQDataset_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config)
        )
    elif dataset_name == "kadid":
        test_dataset = KADIDDataset_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config)
        )
    elif dataset_name == "spaq":
        test_dataset = SPAQDATASET_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config)
        )
    elif dataset_name == "livefb":
        test_dataset = FBLIVEFolder_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config, "livefb")
        )
    elif dataset_name == "bid":
        test_dataset = BIDDATASET_clip(
            dataset_path,
            test_index,
            config.DATA.PATCH_NUM,
            transform=build_clip_transform2(config, "livefb")
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    else:
        sampler_val = IQAPatchDistributedSampler(test_dataset)

    data_loader_val = torch.utils.data.DataLoader(
        test_dataset,
        sampler=sampler_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    return data_loader_val, len(test_dataset)


def CLIP_IQA_build_loader1(config):
    config.defrost()
    dataset_train = build_CLIP_IQA_dataset(config, 1)
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        print("SubsetRandomSampler")
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        print("DistributedSampler")
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    print(len(data_loader_train))
    return data_loader_train


def CLIP_IQA_build_loader2(config):
    config.defrost()
    dataset_train = build_CLIP_IQA_dataset(config, 2)
    print(len(dataset_train))
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        # sampler_train = torch.utils.data.DistributedSampler(
        #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        # )
        sampler_train = RandomIdentitySampler(dataset_train, config.DATA.BATCH_SIZE, config.STAGE2.NUM_INSTANCE)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    print(len(data_loader_train))
    return data_loader_train
