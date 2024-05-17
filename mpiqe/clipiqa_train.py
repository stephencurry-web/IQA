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
from eval_clipiqa import cross_eval

def visual(config, model, data_loader, logger, pre):

    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, target, _, _) in enumerate(data_loader):
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with amp.autocast(enabled=True):
                _, image_feature = model(img)
                target = torch.floor(target)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i.cpu())
                    image_features.append(img_feat.cpu())
        image_labels_list = torch.stack(labels, dim=0).cuda()  # N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = config.DATA.BATCH_SIZE
        num_image = image_labels_list.shape[0]
        i_ter = num_image // batch
        logger.info(f"get total_image_labels:{image_labels_list.shape}")
        logger.info(f"get total_image_features:{image_features_list.shape}")

    combined_features = image_features_list.cpu().detach().numpy()
    combined_labels = image_labels_list.cpu().detach().numpy()

    unique_labels = np.unique(combined_labels)
    print(unique_labels)

    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    reduced_features = tsne.fit_transform(combined_features)

    plt.figure(figsize=(15, 10))

    image_marker = "s"

    # 创建一个颜色映射为每一个独特的标签
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):

        # 图像特征
        plt.scatter(reduced_features[combined_labels == label, 0],
                    reduced_features[combined_labels == label, 1],
                    color=colors[i], marker=image_marker)

    # plt.title('t-SNE Visualization of Image Features')
    # 在显示之前保存图像
    save_path = os.path.join("/home/pws/IQA/global_local/log", config.TAG + str(pre) + "1.png")
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 设置 dpi 为图像的分辨率，bbox_inches='tight' 可以确保图像内容全部保存
    plt.show()
    return


def stage1_train(config, model, data_loader, epochs, optimizer, lr_scheduler, logger):
    max_plcc, max_srcc, max_plcc_c, max_srcc_c = 0.0, 0.0, 0.0, 0.0
    loss_scaler = amp.GradScaler()
    loss_meter = AverageMeter()
    start_time = time.monotonic()
    logger.info("start training")
    logger.info(f"config.ALPHA: {config.ALPHA}, config.BETA: {config.BETA}")

    visual(config, model, data_loader, logger, 0)
    pred_scores_list, gt_scores_list = [], []
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch{epoch} training")
        loss_meter.reset()
        lr_scheduler.step(epoch)
        model.train()
        for n_iter, (img, gt_score, _, _) in enumerate(data_loader):
            optimizer.zero_grad()
            img = img.cuda(non_blocking=True)
            gt_score = gt_score.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                logit, _ = model(img)
                pred_score = 100.0 * logit[:, 0]

            smoothl1_loss = torch.nn.SmoothL1Loss()(pred_score, gt_score)
            loss = smoothl1_loss

            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            loss_meter.update(loss.item(), img.shape[0])
            pred_scores_list = pred_scores_list + pred_score.squeeze().cpu().tolist()
            gt_scores_list = gt_scores_list + gt_score.squeeze().cpu().tolist()
            train_srcc, _ = stats.spearmanr(pred_scores_list, gt_scores_list)
            torch.cuda.synchronize()
            if n_iter % 10 == 0:
                logger.info(f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(data_loader)}] Loss: {loss_meter.avg:.3f} Base Lr: {lr_scheduler._get_lr(epoch)[0]:.2e}, train_srcc: {train_srcc:.3f}")
        
        if epoch % 10 == 0:
            visual(config, model, data_loader, logger, epoch)
            # val_plcc, val_srcc, val_plccc, val_srccc = cross_eval(config, model, logger)
            # logger.info(f"stage 1 validate:{val_plcc}, {val_srcc}, {val_plccc}, {val_srccc}")
            # if val_plcc >= max_plcc:
            #     max_plcc = val_plcc
            #     max_srcc = val_srcc
            #     max_plcc_c = val_plccc
            #     max_srcc_c = val_srccc
            # logger.info(f"stage 1 max:{max_plcc}, {max_srcc}, {max_plcc_c}, {max_srcc_c}")
            #
            # if dist.get_rank() == 0 and config.print:
            #     torch.save(model.state_dict(), os.path.join("/data/pws/backup/output", config.TAG + f"epoch_{epoch}.pth"))


    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    logger.info("Stage1 running time: {}".format(total_time))
    return max_plcc, max_srcc, max_plcc_c, max_srcc_c
