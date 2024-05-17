import argparse
import datetime
import json
import logging
import os
import pickle
import random
import time

import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import AverageMeter  # accuracy
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from IQA import IQA_build_loader
from config import get_config
from logger import create_logger
from lr_scheduler import build_stage1_scheduler, WarmupMultiStepLR, build_stage2_scheduler
from models import Modify_CLIP, load_clip_to_cpu
from IQA import CLIP_IQA_build_loader1, CLIP_IQA_build_loader2
from optimizer import make_optimizer_1stage, make_optimizer_2stage
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
)
from stage1_train import stage1_train


def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--alpha", type=float, help="balance loss")
    parser.add_argument("--beta", type=float, help="balance loss")
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Using tensorboard to track the process",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used (deprecated!)",
    )
    parser.add_argument(
        "--output",
        default="log",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use torchinfo to show the flow of tensor in model",
    )
    parser.add_argument(
        "--repeat", action="store_true", help="Test model for publications"
    )
    parser.add_argument("--rnum", type=int, help="Repeat num")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--depth", type=int, help="Prompt depth")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--token", type=int)
    parser.add_argument("--prompt", type=int)
    parser.add_argument("--scene", action="store_true")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--only_smooth", action="store_true")
    parser.add_argument("--data_percent", type=float)
    parser.add_argument("--print", action="store_true")
    local_rank = int(os.environ["LOCAL_RANK"])
    args, unparsed = parser.parse_known_args()

    config = get_config(args, local_rank)
    return args, config


if __name__ == "__main__":
    args, config = parse_option()
    
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier()

    if args.repeat:
        assert args.rnum > 1
        num = args.rnum
    else:
        num = 1
    logger = logging.getLogger(name=f"{config.MODEL.NAME}")
    base_path = config.OUTPUT

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger.info(f"output:{config.OUTPUT}")
    create_logger(
        logger,
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.MODEL.NAME}",
    )

    logger.info(f"config:{config}")
    total_cc = []
    rnum = config.RNUM
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    start_time = time.time()
    for i in range(rnum):
        if config.print and i != 7:
            continue
        logger.info(f"training num:{i}")
        if rnum > 1:
            config.defrost()
            config.OUTPUT = os.path.join(base_path, str(i))
            config.EXP_INDEX = i + 1
            config.SET.TRAIN_INDEX = None
            config.SET.TEST_INDEX = None
            config.freeze()
        # random.seed(None)
        os.makedirs(config.OUTPUT, exist_ok=True)

        filename = f"{config.SEED}_sel_num{i}.data"
        if dist.get_rank() == 0:
            sel_path = os.path.join(config.SEL_PATH, filename)
            if not os.path.exists(sel_path):
                logger.info("new sel_num")
                sel_num = list(range(0, config.SET.COUNT))
                random.shuffle(sel_num)
                with open(os.path.join(config.SEL_PATH, filename), "wb") as f:
                    pickle.dump(sel_num, f)
                del sel_num
            else:
                logger.info("old sel_num")
        dist.barrier()

        with open(os.path.join(config.SEL_PATH, filename), "rb") as f:
            sel_num = pickle.load(f)

        config.defrost()
        config.SET.TRAIN_INDEX = sel_num[0: int(round(config.data_percent * len(sel_num)))]
        config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        # config.SET.TRAIN_INDEX = sel_num
        logger.info(f"config.SET.TRAIN_INDEX:{config.SET.TRAIN_INDEX}")
        logger.info(f"config.SET.TEST_INDEX:{config.SET.TEST_INDEX}")
        config.freeze()

        
        cudnn.benchmark = True

        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        model = Modify_CLIP(config)
        # summary(model)
        # if i == 0:
        #     logger.info(str(model))
        # params = model.state_dict()  # 获取模型的状态字典
        # print(params.keys())  # 打印模型的参数键
        # print(params['deep_prompt_embeddings'])  # 打印 deep_prompt_embeddings 参数的值
        model.cuda()
        # model_without_ddp = model
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model,
        #     device_ids=[config.LOCAL_RANK],
        #     broadcast_buffers=False,
        #     find_unused_parameters=True,
        # )

        data_loader_train1 = CLIP_IQA_build_loader1(config)
        optimizer_1stage = make_optimizer_1stage(config, model, logger)

        scheduler_1stage = build_stage1_scheduler(optimizer_1stage, num_epochs=config.STAGE1.EPOCHS, lr_min=config.STAGE1.LR_MIN,
                                                  warmup_lr_init=config.STAGE1.WARMUP_LR, warmup_t=config.STAGE1.WARMUP_EPOCHS)
        
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")

        val_plcc, val_srcc, val_plccc, val_srccc = stage1_train(config, model, data_loader_train1, config.STAGE1.EPOCHS, optimizer_1stage, scheduler_1stage, logger)

        end_time = time.time()
        spend_time = (end_time - start_time) / (i+1)
        logger.info(f"spend_time:{spend_time}")
    logger.handlers.clear()