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

def visual(config, model, data_loader, logger, pre):

    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, target, _, _) in enumerate(data_loader):
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with amp.autocast(enabled=True):
                _, image_feature = model(img, eval=True)
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
    save_path = os.path.join("/home/pws/IQA/global_local/log", config.TAG + str(pre) + ".png")
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
    loss1 = SupConLoss()
    loss2 = ImSupConLoss()

    logger.info(f"config.ALPHA: {config.ALPHA}, config.BETA: {config.BETA}")

    visual(config, model, data_loader, logger, 0)

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

            # scene2 = scene2.cuda(non_blocking=True)
            # scene3 = scene3.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                pred_score, logits_global, logits_local = model(img)
            
            # loss_i2t_global = loss1(cls_features, global_features, scene_num, scene_num)
            # loss_t2i_global = loss1(global_features, cls_features, scene_num, scene_num)
            # global_loss = loss_i2t_global + loss_t2i_global
            #
            # loss_i2t_local = loss2(cls_features, local_features, distortion_num, distortion_num)
            # loss_t2i_local = loss2(local_features, cls_features, distortion_num, distortion_num)
            # local_loss = loss_i2t_local + loss_t2i_local
                
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
            
            if config.only_smooth:
                loss = smoothl1_loss
            else:
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
