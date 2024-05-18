import logging
import os
import time
import random
import torch.cuda
import numpy as np
import torch
import time
from datetime import timedelta
import torch.nn as nn
from timm.utils import AverageMeter  # accuracy
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F

from loss import SupConLoss, Fidelity_Loss, Fidelity_Loss_distortion, Multi_Fidelity_Loss
from scipy import stats
from loss import loss_quality
from IQA import build_dataloader, get_labels
from labelsmooth_loss import CrossEntropyLabelSmooth


def get_dataloader(config, dataset_name, logger):
    base_path = "/data/IQA-Dataset"
    if dataset_name == config.DATA.DATASET:
        dataset_path = config.DATA.DATA_PATH
        batch_size = config.DATA.BATCH_SIZE
        test_index = config.SET.TEST_INDEX
        logger.info(f"{dataset_name}:{test_index}")
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "live":
        dataset_path = os.path.join(base_path, "live/databaserelease2")
        count = 29
        batch_size = 12
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "csiq":
        dataset_path = os.path.join(base_path, "CSIQ")
        count = 30
        batch_size = 12
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "tid2013":
        dataset_path = os.path.join(base_path, "tid2013")
        count = 25
        batch_size = 48
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "livec":
        dataset_path = os.path.join(base_path, "ChallengeDB_release")
        count = 1162
        batch_size = 16
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "koniq":
        dataset_path = os.path.join(base_path, "koniq-10k")
        count = 10073
        batch_size = 128
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "kadid":
        dataset_path = os.path.join(base_path, "kadid")
        count = 81
        batch_size = 128
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "spaq":
        dataset_path = os.path.join(base_path, "SPAQ")
        count = 11125
        batch_size = 128
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
    elif dataset_name == "livefb":
        dataset_path = os.path.join(base_path, "liveFB")
        count = 39810
        batch_size = 128
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        test_index = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        logger.info(test_index)
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
        

def cross_eval(config, model, logger):

    if config.DATA.DATASET == "live":
        test_dataset = ["live", "csiq"]
    elif config.DATA.DATASET == "csiq":
        test_dataset = ["csiq", "live"]
    elif config.DATA.DATASET == "tid2013":
        test_dataset = ["tid2013", "kadid"]
    elif config.DATA.DATASET == "kadid":
        test_dataset = ["kadid"]
    elif config.DATA.DATASET == "livec":
        test_dataset = ["livec", "koniq"]
    elif config.DATA.DATASET == "koniq":
        test_dataset = ["koniq", "livec"]
    elif config.DATA.DATASET == "spaq":
        test_dataset = ["spaq", "livefb"]
    elif config.DATA.DATASET == "livefb":
        test_dataset = ["livefb", "livec"]
    elif config.DATA.DATASET == "bid":
        test_dataset = ["bid"]
    result = []
    model.eval()
    with torch.no_grad():
        for idx, dataset in enumerate(test_dataset):
            val_loader, val_len = get_dataloader(config, dataset, logger)
            # total_test = np.zeros((5, 2))
            # for test_num in range(5):
            temp_pred_scores = []
            temp_gt_scores = []
            for n_iter, (img, labels, _, _) in enumerate(val_loader):
                img = img.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with amp.autocast(enabled=True):
                    preds = model(img, eval=True)
                temp_pred_scores.append(preds.view(-1))
                temp_gt_scores.append(labels.view(-1))
                # pred_scores = pred_scores + preds.squeeze().cpu().tolist()
                # gt_scores = gt_scores + labels.squeeze().cpu().tolist()

            pred_scores = torch.cat(temp_pred_scores)
            gt_scores = torch.cat(temp_gt_scores)
            # For distributed parallel, collect all data and then run metrics.
            if torch.distributed.is_initialized():
                preds_gather_list = [
                    torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
                ]
                torch.distributed.all_gather(preds_gather_list, pred_scores)
                gather_preds = torch.cat(preds_gather_list, dim=0)[:val_len]
                gather_preds = (
                    (gather_preds.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
                ).squeeze()
                grotruth_gather_list = [
                    torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
                ]
                torch.distributed.all_gather(grotruth_gather_list, gt_scores)
                gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_len]
                gather_grotruth = (
                    (gather_grotruth.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
                ).squeeze()
                final_preds = gather_preds.cpu().tolist()
                final_grotruth = gather_grotruth.cpu().tolist()
            # logger.info(f"final_preds:{final_preds[0:100]}")
            # logger.info(f"final_grotruth:{final_grotruth[0:100]}")
            test_srcc, _ = stats.spearmanr(final_preds, final_grotruth)
            test_plcc, _ = stats.pearsonr(final_preds, final_grotruth)
            result.append(test_plcc)
            result.append(test_srcc)
    if len(result) < 4:
        result.append(0.0)
        result.append(0.0)
    return result