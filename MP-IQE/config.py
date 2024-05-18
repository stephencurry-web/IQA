import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
# K10K: /data/qgy/IQA-Dataset/koinq-10k
# LIVEC: /home/qinguanyi/dataset/IQA/ChallengeDB_release
_C.DATA.DATA_PATH = "/home/pws/IQA/dataset/tid2013"
# Dataset name
_C.DATA.DATASET = "tid2013"
# Aug images
_C.DATA.PATCH_NUM = 25
# Input image size
_C.DATA.IMG_SIZE = 224
# Random crop image size
_C.DATA.CROP_SIZE = (224, 224)
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = "part"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
_C.DATA.SCENE_NUM_CLASSES = 9
_C.DATA.DIST_NUM_CLASSES = 11
_C.DATA.H_RESOLUTION = 224
_C.DATA.W_RESOLUTION = 224

# -----------------------------------------------------------------------------
# SET settings
# -----------------------------------------------------------------------------
_C.SET = CN()
# K10K: 10073 LIVEC: 1162
_C.SET.COUNT = 10073
_C.SET.TRAIN_INDEX = None
_C.SET.TEST_INDEX = None
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "swin"
_C.MODEL.BACKBONE = "ViT-B-16"
# Model name
_C.MODEL.NAME = "swin_tiny_patch4_window7_224"
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ""
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ""
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.NUM_TOKENS = 4
_C.MODEL.DROPOUT = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.SCORE_HEAD = 16
_C.MODEL.SWIN.SCORE_DIM = 32
_C.MODEL.SWIN.PRETRAINED = False
_C.MODEL.SWIN.PRETRAINED_MODEL_PATH = ""

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.0
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.0
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True


# ConvNeXt parameters
_C.MODEL.CONV_NEXT = CN()
_C.MODEL.CONV_NEXT.IN_CHANS = 3
_C.MODEL.CONV_NEXT.DIM = [96, 192, 384, 768]
_C.MODEL.CONV_NEXT.DEPTHS = [3, 3, 9, 3]
_C.MODEL.CONV_NEXT.LAYER_SCALE_INIT = 1e-6
_C.MODEL.CONV_NEXT.HEAD_INIT_SCALE = 1.0
_C.MODEL.CONV_NEXT.SCORE_HEAD = 16
_C.MODEL.CONV_NEXT.SCORE_DIM = 32
_C.MODEL.CONV_NEXT.PRETRAINED = False
_C.MODEL.CONV_NEXT.PRETRAINED_MODEL_PATH = ""

# DeiT III
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.EMBED_DIM = 384
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 6
_C.MODEL.VIT.MLP_RATIO = 4
_C.MODEL.VIT.QKV_BIAS = True
_C.MODEL.VIT.PRETRAINED = True
_C.MODEL.VIT.PRETRAINED_MODEL_PATH = ""

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.STAGE1 = CN()
_C.STAGE1.WEIGHT_DECAY = 0.001
_C.STAGE1.BASE_LR = 3.0e-4
_C.STAGE1.EPOCHS = 30
_C.STAGE1.LR_MIN = 0.000002
_C.STAGE1.WARMUP_LR = 0.00001
_C.STAGE1.WARMUP_EPOCHS = 5

_C.STAGE2 = CN()
_C.STAGE2.WEIGHT_DECAY = 0.0001
_C.STAGE2.BASE_LR = 5.0e-6
_C.STAGE2.EPOCHS = 60
_C.STAGE2.WARMUP_EPOCHS = 5
_C.STAGE2.WARMUP_LR = 2e-7
# _C.STAGE2.LR_MIN = 0.000016
# _C.STAGE2.WARMUP_LR = 0.00001
# _C.STAGE2.WARMUP_EPOCHS = 5
# _C.STAGE2.DECAY_EPOCHS = 5
# _C.STAGE2.DECAY_RATE = 0.9
_C.STAGE2.GAMMA = 0.1
# decay step of learning rate
_C.STAGE2.STEPS = (30, 50)
# warm up factor
_C.STAGE2.WARMUP_FACTOR = 0.1
#  warm up iters
_C.STAGE2.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
_C.STAGE2.WARMUP_METHOD = "linear"
_C.STAGE2.NUM_INSTANCE = 4
_C.STAGE2.ALPHA = 0.001
_C.STAGE2.BETA = 0.1
_C.STAGE2.MARGIN = 0.3

_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 9
_C.TRAIN.WARMUP_EPOCHS = 3
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 3.0e-4
_C.TRAIN.WARMUP_LR = 2.0e-7
_C.TRAIN.MIN_LR = 2.0e-7
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.COOP_N_CTX = 8
_C.TRAIN.COOP_CSC = False
_C.TRAIN.COOP_CLASS_TOKEN_POSITION = "end"
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = "pixel"
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ""
# Path to output folder, overwritten by command line argument
_C.OUTPUT = "log"
_C.SEL_PATH = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Use tensorboard to track the trainning log
_C.TENSORBOARD = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Use torchinfo to show the flow
_C.DEBUG_MODE = False
# Repeat exps for publications
_C.EXP_INDEX = 0
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.RNUM = 5
_C.DEPTH = 11
_C.ALPHA = 1.0
_C.BETA = 0.0
_C.num_scene = 9
_C.num_dist = 11
_C.visual = False
_C.scene = False
_C.dist = False
_C.data_percent = 0.8
_C.print = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args, local_rank):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.alpha:
        config.STAGE2.ALPHA = args.alpha
    if args.beta:
        config.STAGE2.BETA = args.beta
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == "O0":
            config.AMP_ENABLE = False
    if args.disable_amp:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.tensorboard:
        config.TENSORBOARD = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.debug:
        config.DEBUG_MODE = True
    if args.rnum:
        config.RNUM = args.rnum
    if args.depth:
        config.DEPTH = args.depth
    if args.alpha:
        config.ALPHA = args.alpha
    if args.beta:
        config.BETA = args.beta
    if args.seed:
        config.SEED = args.seed
    if args.epoch:
        config.STAGE1.EPOCHS = args.epoch
    if args.visual:
        config.visual = True
    if args.scene:
        config.scene = args.scene
    if args.dist:
        config.dist = args.dist
    if args.token:
        config.MODEL.NUM_TOKENS = args.token
    if args.prompt:
        config.TRAIN.COOP_N_CTX = args.prompt
    if args.data_percent:
        config.data_percent = args.data_percent
    if args.print:
        config.print = args.print
    # set local rank for distributed training
    config.LOCAL_RANK = local_rank

    # output folder
    # config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    # output folder
    config.SEL_PATH = os.path.join(config.OUTPUT, config.DATA.DATASET)
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)

    config.freeze()


def get_config(args, local_rank):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args, local_rank)

    return config
