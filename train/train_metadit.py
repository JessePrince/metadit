"""
Training Script for project ColorNet
"""

import os
import sys
sys.path.append(os.getcwd())

from transformers import (
    set_seed,
    
)
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field, asdict
from loggers import WrappedLogger
from model.dit import metadit_s, metadit_b, metadit_l
from datapipe import FreeFormDataset
from engine import Trainer
from utils import save_json
from torch.utils.data import random_split
from diffusion import create_diffusion
import torch
from model.dit import DIT_MODEL

logger = WrappedLogger(__name__)
rank = int(os.getenv("LOCAL_RANK", -1))


@dataclass
class TrainArgs:
    # Training
    num_epoch: int = field(default=10)
    batch_size: int = field(default=1024)
    # optimization
    optimizer: str = field(default="adamw", metadata={"choices": ["adamw", "adam"]})
    lr: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.01)
    weight_decay: float = field(default=0.01)
    use_checkpointing: bool = field(default=False)
    # data
    data_path: str = field(default="/root/autodl-tmp/colornet/sim_data/crystal_50000.json")
    val_path: str = field(default="root/autodl-tmp")
    num_workers: int = field(default=8)
    # ckpt
    save_dir: str = field(default="ckpt/metadiff-1")
    save_strategy: str = field(default="epoch", metadata={"choices": ["epoch", "step"]})
    save_step: str = field(default=1000)
    save_epoch: str = field(default=1)
    save_total_limit: int = field(default=3)
    # log
    log_to: str = field(default="tensorboard")
    tb_logdir: str = field(default="/root/tf-logs")
    # eval
    eval_strategy: str = field(default="epoch")
    eval_step: int = field(default=1000)

@dataclass
class ModelArgs:
    model_type: str = field(default="metadit_s")
    num_latent_size: int = field(default=1024)
    high_res_spec: bool = field(default=False)
    condition_channel: int = field(default=52)
    pretrain_encoder: str = field(default=None)
    

def parse_args():
    parser = HfArgumentParser((TrainArgs, ModelArgs))
    train_args, model_args = parser.parse_args_into_dataclasses()
    
    return train_args, model_args

def profile_everything(path, train_args, model_args, model):
    profile = {}
    profile["Train Args"] = asdict(train_args)
    profile["Model Args"] = asdict(model_args)
    profile["Model Structure"] = str(model).split("\n")
    profile["Activated Param"] = [name for name, param in model.named_parameters() if param.requires_grad]
    profile["Total Param"] = sum([param.nelement() for param in model.parameters()])
    profile["Trainable Param"] = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    
    save_json(path, profile)

def main(
    train_args: TrainArgs,
    model_args: ModelArgs
):
    logger.info(f"Training script for VAE", on_rank0=True)
    if rank != -1:
        set_seed(0 + rank)
    
    diffusion = create_diffusion("", learn_sigma=False)
    model = DIT_MODEL[model_args.model_type](diffusion=diffusion, condition_channel=model_args.condition_channel)
    if model_args.pretrain_encoder is not None:
        ckpt = torch.load(model_args.pretrain_encoder)
        ckpt = {k.split("context_encoder.")[1]: v for k, v in ckpt.items()}
        model.y_embedder.encoder.load_state_dict(ckpt)
        model.y_embedder.encoder.requires_grad_(False)
    logger.info(str(model), on_rank0=True)
    if rank in [0, -1]:
        if not os.path.exists(train_args.save_dir):
            os.makedirs(train_args.save_dir)
        profile_everything(os.path.join(train_args.save_dir, "profile.json"), train_args, model_args, model)
    
    
    train_set = FreeFormDataset(path=train_args.data_path)
    val_set = FreeFormDataset(path=train_args.val_path)
    trainer = Trainer(
        model,
        train_set,
        val_set,
        train_args.batch_size,
        train_args.weight_decay,
        train_args.lr,
        train_args.warmup_ratio,
        train_args.num_workers,
        "cosine",
        False,
        train_args.num_epoch,
        train_args.log_to,
        train_args.save_dir,
        train_args.tb_logdir,
        train_args.save_strategy,
        train_args.save_step,
        train_args.save_epoch,
        train_args.eval_strategy,
        train_args.eval_step,
        train_args.save_total_limit
    )
    trainer.train()
    

if __name__ == "__main__":
    train_args, model_args = parse_args()
    main(train_args, model_args)
    