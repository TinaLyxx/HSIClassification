import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import transforms,datasets

from .data_load import *
from model import Main_Model


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'v1' or model_type == 'v2':
        model = Main_Model(
            in_channels=config.DATA.CHANNELS,
            embed_dim=config.MODEL.EMBED_DIMENSION,
            num_classes=config.MODEL.NUM_CLASSES,
            patch_size=config.DATA.PATCH_SIZE,
            group_num=config.MODEL.GROUP_NUM,
            group_norm=config.MODEL.GROUP_NORM,
            basic_version=config.MODEL.TYPE,
            depths=config.MODEL.DEPTHS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            num_head=config.MODEL.ATTENTION_HEAD,
            hidden_len=config.MODEL.HIDDEN_LEN,
            spe_headdim=config.MODEL.SPE_HEAD_DIM,
            spa_query_len=config.MODEL.SPA_QUERY_LEN,
            head_typ=config.MODEL.HEAD_TYPE,
            v2_if_resize=config.MODEL.V2_RESIZE
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


def build_dataset(config):
    dataset = config.DATA.DATASET
    val_size = config.DATA.TEST_SIZE
    train_size = config.DATA.TRAIN_SIZE
    sample_mode = config.DATA.SAMPLE_MODE

    image, label,_,_= load_data(dataset, config)
    train_label, test_label = sample_gt(label, train_size, sample_mode)
    val_label, train_label = sample_gt(train_label, val_size, sample_mode)

    train_dataset = HyperX(image, train_label, config)
    val_dataset = HyperX(image, val_label, config)
    test_dataset = HyperX(image, test_label, config)

    return train_dataset, val_dataset, test_dataset, test_label

def build_loader(config):
    dataset_train, dataset_val, dataset_test, test_label = build_dataset(config)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    if dataset_train is not None:
        sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=True)
    if dataset_val is not None:
        if config.TEST.SEQUENTIAL:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, shuffle=False)
    if dataset_test is not None:   
        if config.TEST.SEQUENTIAL:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, shuffle=False)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_test is not None else None
    
    return dataset_train, dataset_val, dataset_test, test_label, data_loader_train, data_loader_val, data_loader_test

def build_loader_single(config):
    dataset_train, dataset_val, dataset_test, test_label = build_dataset(config)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False) if dataset_test is not None else None
    
    return dataset_train, dataset_val, dataset_test, test_label, data_loader_train, data_loader_val, data_loader_test

    





