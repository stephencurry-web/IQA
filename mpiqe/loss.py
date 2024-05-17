from __future__ import print_function

import torch
import torch.nn as nn
EPS = 1e-2
esp = 1e-8


def ranking_loss(y_pred, y_true, scale_ = 2.0, margin_ = 1):

    y_true = torch.zeros(y_pred.size()).scatter_(1, y_true.unsqueeze(1).data.cpu(), 1).to(y_pred.device)
    y_pred *= scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)


def ranking_loss_multi(y_pred, y_true, y_true1, y_true2, scale_ = 2.0, margin_ = 1):

    y_true_one_hot = torch.zeros_like(y_pred)

    # 对每个 y_true 进行 one-hot 编码并累加
    for y in [y_true, y_true1, y_true2]:
        # 忽略索引为-1或其他无效值的情况，这些值不代表任何类别
        valid_indices = y >= 0
        y_valid = y[valid_indices].unsqueeze(1)  # 保留有效索引并增加一个维度以适应 scatter_ 的需求

        # 使用 scatter_ 在正确的索引位置上放置 1，并累加到 y_true_one_hot 上
        y_true_one_hot[valid_indices] = y_true_one_hot[valid_indices].scatter_add_(1, y_valid, torch.ones_like(y_valid, dtype=y_true_one_hot.dtype) )

    # 确保 y_true_one_hot 中的值不超过 1，因为一个类别只需标记一次
    y_true = y_true_one_hot.clamp(max=1)

    # 将 y_true_one_hot 转移到与 y_pred 相同的设备上
    y_true = y_true.to(y_pred.device)

    # y_true = torch.zeros(y_pred.size()).scatter_(1, y_true.unsqueeze(1).data.cpu(), 1).to(y_pred.device)

    y_pred *= scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)

class SupConLoss(nn.Module):
    def __init__(self):
        super(SupConLoss, self).__init__()
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to("cuda")
        # print(image_features.shape)
        # print(text_features.shape)
        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        # print(logits)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()
        return loss

class BCELosswithLogits(nn.Module):
      def __init__(self, pos_weight=1, reduction='mean'):
          super(BCELosswithLogits, self).__init__()
          self.pos_weight = pos_weight
          self.reduction = reduction

      def forward(self, logits, target):
          # logits: [N, *], target: [N, *]
          logits = F.sigmoid(logits)
          loss = - self.pos_weight * target * torch.log(logits) - \
                (1 - target) * torch.log(1 - logits)
          if self.reduction == 'mean':
              loss = loss.mean()
          elif self.reduction == 'sum':
              loss = loss.sum()
          return loss
      
      
def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bce_loss

    weighted_loss = alpha * loss
    loss = torch.sum(weighted_loss)
    loss /= torch.sum(labels)
    return loss


class ImSupConLoss(nn.Module):
    def __init__(self):
        super(ImSupConLoss, self).__init__()
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets):
        
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to("cuda")
        # print(image_features.shape)
        # print(text_features.shape)
        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        # print(logits)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()
        return loss


import torch.nn.functional as F


def InfoNCE_loss(text_features, image_features, t_labels, i_targets, temperature=0.5):
    """
    Infonce loss function for text and image features.

    Args:
        text_features (torch.Tensor): text feature tensor, shape [batch_size, feature_dim].
        image_features (torch.Tensor): image feature tensor, shape [batch_size, feature_dim].
        t_labels (torch.Tensor): text labels tensor, shape [batch_size,].
        i_targets (torch.Tensor): image targets tensor, shape [batch_size,].
        temperature (float): temperature parameter for softmax operation. Default: 0.5.

    Returns:
        infonce_loss (torch.Tensor): Infonce loss tensor.
    """

    # Concatenate text and image features
    features = torch.cat([text_features, image_features], dim=0)

    # Create a mask to exclude diagonal elements from the dot product
    batch_size = features.size(0)
    mask = torch.ones(batch_size, batch_size)
    mask = mask.fill_diagonal_(0)

    # Compute cosine similarity matrix
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    sim_matrix = sim_matrix / temperature

    # Compute positive pair loss
    t_i_targets = torch.cat([t_labels, i_targets], dim=0)
    pos_mask = torch.eq(t_i_targets.unsqueeze(1), t_i_targets.unsqueeze(0)).float()
    pos_mask = pos_mask / temperature
    pos_sim = torch.exp(torch.masked_select(sim_matrix, pos_mask.bool()))
    pos_loss = -torch.log(pos_sim / torch.sum(pos_mask, dim=1))

    # Compute negative pair loss
    neg_mask = mask
    neg_sim = torch.exp(torch.masked_select(sim_matrix, neg_mask.bool()))
    neg_loss = -torch.log(torch.sum(neg_sim) / torch.sum(neg_mask))

    # Combine positive and negative losses
    infonce_loss = torch.mean(pos_loss) + neg_loss

    return infonce_loss



class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))

        return torch.mean(loss)


class Fidelity_Loss_distortion(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss_distortion, self).__init__()

    def forward(self, p, g):
        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = torch.sqrt(p_i * g_i + esp)
            loss = loss + loss_i
        loss = 1 - loss
        loss = loss / p.size(1)
        return torch.mean(loss)


class Multi_Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Multi_Fidelity_Loss, self).__init__()

    def forward(self, p, g):

        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = 1 - (torch.sqrt(p_i * g_i + esp) + torch.sqrt((1 - p_i) * (1 - g_i) + esp))
            loss = loss + loss_i
        loss = loss / p.size(1)

        return torch.mean(loss)


def loss_quality(y_pred, y):
    """prediction monotonicity related loss"""
    loss = 0

    #assert y_pred.size(0) > 1  #
    if y_pred.size(0) == 0:
        return
    y_pred = y_pred.unsqueeze(1)
    y = y.unsqueeze(1)

    preds = y_pred - y_pred.t()
    # y_pred = [1,2,3,4,5], preds = [[ 0, -1, -2, -3, -4],
    #  [ 1,  0, -1, -2, -3],
    #  [ 2,  1,  0, -1, -2],
    #  [ 3,  2,  1,  0, -1],
    #  [ 4,  3,  2,  1,  0]]
    gts = y - y.t()

    # signed = torch.sign(gts)

    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    g = 0.5 * (torch.sign(gts) + 1)

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss += torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss


def featureL2Norm(feature):
     epsilon = 1e-6
     norm = torch.pow(torch.sum(torch.pow(feature, 2), 1)+epsilon, 0.5).unsqueeze(1).expand_as(feature)
     return torch.div(feature, norm)

def compute_div(w):
     K, C = w.shape
     w_norm = featureL2Norm(w)
     WWT = torch.matmul(w_norm, w_norm.transpose(0, 1))
     return F.mse_loss(WWT - torch.eye(K).cuda(), torch.zeros(K, K).cuda(), size_average=False) / (K*K)