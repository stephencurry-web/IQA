from torch import optim as optim
import torch

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def make_optimizer_1stage(config, model, logger):
    params = []
    for key, value in model.named_parameters():
        # params.append(value)
        if "prompt_learner" in key:
            logger.info(key)
            params.append(value)
            continue

        if "encoder_proj" in key:
            logger.info(key)
            params.append(value)
            continue

        if "prompt_proj" in key:
            logger.info(key)
            params.append(value)
            continue

        if "prompt_embeddings" in key:
            logger.info(key)
            params.append(value)
            continue

        if "decoder" in key:
            logger.info(key)
            params.append(value)
            continue

        if "logit_scale" in key:
            logger.info(key)
            params.append(value)
            continue
        
        if "adaptive_max_pool" in key:
            logger.info(key)
            params.append(value)
            continue

        # value.requires_grad_(False)
        
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(params, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.STAGE1.BASE_LR, weight_decay=config.STAGE1.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(params, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.STAGE1.BASE_LR, weight_decay=config.STAGE1.WEIGHT_DECAY)
    
    n_params_optimizer = sum(p.numel() for p in params if p.requires_grad)
    logger.info(f"number of params passed to optimizer: {n_params_optimizer}")
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
