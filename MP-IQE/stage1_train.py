import datetime
import time
import torch
from datetime import timedelta
import os
from torch.cuda import amp
import numpy as np
import torch.distributed as dist
from torch.nn import functional as F
from scipy import stats
from timm.utils import AverageMeter  # accuracy
from loss import SupConLoss, ImSupConLoss, Fidelity_Loss, Fidelity_Loss_distortion, Multi_Fidelity_Loss, InfoNCE_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from loss import loss_quality, ranking_loss_multi, ranking_loss
from labelsmooth_loss import CrossEntropyLabelSmooth
from eval import cross_eval

def stage1_train(config, model, data_loader, epochs, optimizer, lr_scheduler, logger):
    max_plcc, max_srcc, max_plcc_c, max_srcc_c = 0.0, 0.0, 0.0, 0.0
    loss_scaler = amp.GradScaler()
    loss_meter = AverageMeter()
    start_time = time.monotonic()
    logger.info("start training")
    logger.info(f"config.ALPHA: {config.ALPHA}, config.BETA: {config.BETA}")

    pred_scores_list, gt_scores_list = [], []
    train_srcc = 0.0
    # val_plcc, val_srcc, val_plccc, val_srccc = cross_eval(config, model, logger)
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch{epoch} training")
        loss_meter.reset()
        lr_scheduler.step(epoch)
        model.train()
        for n_iter, (img, gt_score, scene_num, distortion_num) in enumerate(data_loader):
            optimizer.zero_grad()
            img = img.cuda(non_blocking=True)
            gt_score = gt_score.cuda(non_blocking=True)
            scene_num = scene_num.cuda(non_blocking=True)
            distortion_num = distortion_num.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                pred_score, logits_global, logits_local = model(img)
                
            global_loss, local_loss = 0.0, 0.0

            # multi-label
            if logits_global is not None:
                smooth_loss_global = CrossEntropyLabelSmooth(num_classes=config.DATA.SCENE_NUM_CLASSES)
                global_loss = smooth_loss_global(logits_global, scene_num)
                # global_loss = ranking_loss_multi(logits_global, scene_num, scene2, scene3, scale_=4.0)

            if logits_local is not None:
                smooth_loss_local = CrossEntropyLabelSmooth(num_classes=config.DATA.DIST_NUM_CLASSES)           
                local_loss = smooth_loss_local(logits_local, distortion_num)
                # local_loss = ranking_loss(logits_local, distortion_num, scale_=4.0)

            fidelity_loss = loss_quality(pred_score, gt_score)
            smoothl1_loss = torch.nn.SmoothL1Loss()(pred_score, gt_score)
            loss = global_loss + config.ALPHA * local_loss + config.BETA * smoothl1_loss

            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            loss_meter.update(loss.item(), img.shape[0])
            pred_scores_list = pred_scores_list + pred_score.squeeze().cpu().tolist()
            gt_scores_list = gt_scores_list + gt_score.squeeze().cpu().tolist()
            train_srcc, _ = stats.spearmanr(pred_scores_list, gt_scores_list)
            torch.cuda.synchronize()
            if n_iter % 10 == 0:
                logger.info(f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(data_loader)}] Loss: {loss_meter.avg:.3f}, Global_Loss: {global_loss:.3f}, Local_Loss: {local_loss:.3f} Fidelity_Loss: {fidelity_loss:.3f} Smoothl1_loss:{smoothl1_loss:.3f} Base Lr: {lr_scheduler._get_lr(epoch)[0]:.2e}, train_srcc: {train_srcc:.3f}")

        if epoch % 10 == 0:
            val_plcc, val_srcc, val_plccc, val_srccc = cross_eval(config, model, logger)
            logger.info(f"stage 1 validate:{val_plcc}, {val_srcc}, {val_plccc}, {val_srccc}")
            if val_plcc >= max_plcc:
                max_plcc = val_plcc
                max_srcc = val_srcc
                max_plcc_c = val_plccc
                max_srcc_c = val_srccc
            logger.info(f"stage 1 max:{max_plcc}, {max_srcc}, {max_plcc_c}, {max_srcc_c}")

    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    logger.info("Stage1 running time: {}".format(total_time))
    return max_plcc, max_srcc, max_plcc_c, max_srcc_c
