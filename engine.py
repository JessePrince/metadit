"""
Train engine
"""

import os
import shutil
from loggers import WrappedLogger, WandbLogger, TensorBoardLogger
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
import math
from transformers import get_scheduler
import torch
from datetime import datetime
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset
import pdb
import torch.distributed as dist

logger = WrappedLogger(__name__)

rank = os.environ.get("RANK", -1)

NAME2PRECISION = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

class Trainer():
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        weight_decay: float,
        lr: float,
        warmup_ratio: float,
        num_workers: int,
        lr_scheduler: str,
        enable_ema: bool,
        num_epoch: int,
        log_to: str,
        work_dir: str,
        tb_logdir: str,
        save_strategy: str,
        save_step: int,
        save_epoch: int,
        eval_strategy: str,
        eval_step: int,
        save_total_limit: int
    ):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.save_strategy = save_strategy
        self.save_total_limit = save_total_limit
        self.save_step = save_step
        self.save_epoch = save_epoch
        self.eval_strategy = eval_strategy
        self.eval_step = eval_step
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.train_state = {
            "step": 0,
            "epoch": 0
        }
        self.epoch_end = False

        self.model = model
        optimizer_cls = (
            optim.AdamW
            if self.accelerator.state.deepspeed_plugin is None
            or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )
        optimizer = optimizer_cls(self.partition_param(weight_decay), lr=lr)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
        
        # important: accelerator doesn't count samples that are discarded due to DP
        step_per_epoch_rounded = math.floor(len(train_loader) / self.accelerator.num_processes) * self.accelerator.num_processes

        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            lr_scheduler = get_scheduler(
                name=lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=int(step_per_epoch_rounded*num_epoch*warmup_ratio),
                num_training_steps=step_per_epoch_rounded*num_epoch,
            )
        else:
            lr_scheduler = DummyScheduler(
                optimizer, total_num_steps=step_per_epoch_rounded*num_epoch, warmup_num_steps=step_per_epoch_rounded*num_epoch
            )
        
        (
            self.model_wrapped,
            self.train_loader,
            self.optimizer,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model, 
            train_loader,
            optimizer,
            lr_scheduler
        )

        self.model_precision = self.get_model_precision()
        self.epoch_unit = 1 / len(self.train_loader)
        
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False, persistent_workers=True)
            self.val_loader = self.accelerator.prepare(val_loader)
        else:
            self.val_loader = None
            
        self.should_log = {}
        
        if self.accelerator.is_main_process:
            if log_to == "wandb":
                self.online_logger = WandbLogger(work_dir)
            elif log_to == "tensorboard":
                self.online_logger = TensorBoardLogger(os.path.join(tb_logdir, os.path.split(work_dir)[-1]))
            elif log_to == "none":
                self.online_logger = None
            else:
                raise ValueError(f"Unrecognized logger {log_to}")
        
        self.work_dir = work_dir
        if self.accelerator.is_main_process:
            if not os.path.exists(work_dir):
                os.makedirs(work_dir, exist_ok=True)
            
            
    def partition_param(self, weight_decay: float):
        no_decay = ["bias", "LayerNorm.weight", "GroupNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return optimizer_grouped_parameters
    
    def move_to_(self, data: dict):
        kwargs = {"device": self.accelerator.device, "dtype": self.model_precision}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(**kwargs)
        return data
        
    def train_one_epoch(self, data_loader: DataLoader):
        self.model_wrapped.train()
        self.accelerator.wait_for_everyone()
        self.optimizer.zero_grad()
        # Call self.accelerator.free_memory() results in self.accelerator.deepspeed_engine_wrapped = None!!!
        torch.cuda.empty_cache()
        self.epoch_end = False
        self.should_log.clear()
        for idx, data in enumerate(data_loader):
            data = self.move_to_(data)
            output = self.model_wrapped(**data)
            
            metrics = {k: self.accelerator.reduce(v.detach(), reduction="mean") for k, v in output.items() if "loss" in k}
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["Memory"] = f"{torch.cuda.max_memory_allocated()/1e9} GB"
                
            self.accelerator.backward(output["loss"])
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.train_state["step"] += 1
            self.train_state["epoch"] += self.epoch_unit
            if self.accelerator.is_main_process:
                self.log(self.train_state, prefix="train")
                self.log(metrics, prefix="train")
                self.display_log(self.pbar)
                self.pbar.update(1)
            
            self.maybe_save()
            self.maybe_eval()
        
        self.epoch_end = True
        self.maybe_save()
        self.maybe_eval()
        
    def eval_loop(self):
        assert self.val_loader is not None, "Cannot eval without eval loader"
        self.model_wrapped.eval()
        self.accelerator.wait_for_everyone()
        logger.info(f"Entering Evaluation...")
        
        # Store all per-batch outputs locally on each process first
        all_step_outputs = []
        all_batch_sizes = [] 
        with torch.no_grad():
            for idx, data in enumerate(self.val_loader):
                # No need for move_to_; accelerator handles device placement of the loader
                data = self.move_to_(data)
                batch_size = next(iter(data.values())).size(0)
                output = self.model_wrapped(**data)
                
                # Keep only the tensors needed for metrics
                step_outputs = {k: v.detach() for k, v in output.items() if "loss" in k}
                all_step_outputs.append(step_outputs)
                all_batch_sizes.append(batch_size)
                
        # Gathers all dictionaries from all processes. Accelerator handles the logic.
        # This correctly handles cases where processes have a different number of batches.
        
        # dim 1 steps, dim 2 dict , tensor with shape of world size
        eval_metrics = self.accelerator.gather_for_metrics(all_step_outputs)
        # flattened steps*world_size
        batch_sizes = self.accelerator.gather_for_metrics(all_batch_sizes)

        # Now, calculate the true mean on the main process
        if self.accelerator.is_main_process:
            average_metrics = {}
            flattened_eval_metrics = {}
            keys = eval_metrics[0].keys()
            for k in keys:
                flattened_eval_metrics[k] = [v for m in eval_metrics for v in m[k]]

            # Ensure they have the same length
            assert len(next(iter(flattened_eval_metrics.values()))) == len(batch_sizes), \
                f"Mismatch in gathered metrics and batch sizes {len(next(iter(flattened_eval_metrics.values())))}, {len(batch_sizes)}."
            
            for k in keys:
                weighted_sum = 0.0
                total_samples = 0
                
                for i, v in enumerate(flattened_eval_metrics[k]):
                    batch_size = batch_sizes[i]
                    weighted_sum += v.item() * batch_size
                    total_samples += batch_size
                
                if total_samples > 0:
                    average_metrics[k] = weighted_sum / total_samples
                else:
                    average_metrics[k] = 0.0 # Handle case with no samples
                
            # Log the final, correct average metrics
            logger.info(f"Average Metrics: {average_metrics}")
            self.log(average_metrics, prefix="eval")

        torch.cuda.empty_cache()

        
    # def eval_loop(self):
    #     assert self.val_loader is not None, "Cannot eval without eval loader"
    #     self.model_wrapped.eval()
    #     self.should_log.clear()
    #     self.accelerator.wait_for_everyone()
    #     logger.info(f"Entering Evaluation...")
    #     torch.cuda.empty_cache()
    #     if self.accelerator.is_main_process:
    #         eval_pbar = tqdm(total=len(self.val_loader), desc="Evaluation")
        
    #     all_metrics = []
    #     with torch.no_grad():
    #         for idx, data in enumerate(self.val_loader):
    #             data = self.move_to_(data)
    #             output = self.model_wrapped(**data)
                
    #             metrics = {k: self.accelerator.reduce(v.detach(), reduction="mean") for k, v in output.items() if "loss" in k}
    #             metrics["Memory"] = f"{torch.cuda.max_memory_allocated()/1e9} GB"
    #             if self.accelerator.is_main_process:
    #                 self.log(metrics, prefix="eval", to_online_logger=False)
    #                 self.display_log(eval_pbar)
    #                 eval_pbar.update(1)
                
    #             all_metrics.append({k: v for k, v in metrics.items() if "loss" in k})
                
    #         keys = all_metrics[0].keys()
    #         average_metrics = {}
    #         for k in keys:
    #             # not accurate if drop_last=False, FIXME
    #             average_metrics[k] = sum([metric[k] for metric in all_metrics]) / len(all_metrics)
                
    #         if self.accelerator.is_main_process:
    #             eval_pbar.write(f"Average Metrics: {average_metrics}")
    #             self.log(average_metrics, prefix="eval")
            
    def get_model_param_count(self, trainable_only: bool):
        if trainable_only:
            count = sum([param.nelement() for param in self.model.parameters() if param.requires_grad])
        else:
            count = sum([param.nelement() for param in self.model.parameters()])
            
        return count
    
    def get_model_precision(self):
        for param in self.model_wrapped.parameters():
            if param.dtype in [torch.bfloat16, torch.float16]:
                return param.dtype
            
        return next(self.model_wrapped.parameters()).dtype
            
    def train(self, resume=True):
        logger.info("***** Running training *****", on_rank0=True)
        logger.info(f"  Num examples = {int(len(self.train_loader))*self.batch_size*self.accelerator.num_processes}", on_rank0=True)
        logger.info(f"  Num Epochs = {int(self.num_epoch)}", on_rank0=True)
        logger.info(f"  Instantaneous batch size per device = {self.batch_size}", on_rank0=True)
        if self.accelerator.num_processes > 1:
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.batch_size*self.accelerator.num_processes}", on_rank0=True)
        logger.info(f"  Total optimization steps = {len(self.train_loader)*self.num_epoch}", on_rank0=True)
        logger.info(f"  Number of trainable parameters = {self.get_model_param_count(trainable_only=True)}", on_rank0=True)
        logger.info(f"  Training with precision = {self.accelerator.mixed_precision}", on_rank0=True)
        if self.accelerator.is_main_process:
            self.pbar = tqdm(total=len(self.train_loader)*self.num_epoch, desc="Train")
            
        starting_epoch = 0
        if resume and len(os.listdir(self.work_dir)) > 0:
            saved_files = os.listdir(self.work_dir)
            # pdb.set_trace()
            if any(["step" in f or "epoch" in f for f in saved_files]):
                saved_ckpts = os.listdir(self.work_dir)
                saved_ckpts = [file for file in saved_ckpts if ("epoch" in file) or ("step" in file)]
                saved_ckpts.sort(key=lambda x: int(x.split("_")[1]))
                latest = saved_ckpts[-1]
                complete_dir = os.path.join(self.work_dir, latest)
                self.accelerator.load_state(complete_dir)
                logger.info(f"Resume from {complete_dir}")
                path = os.path.basename(complete_dir)
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", ""))
                    resume_step = None
                    completed_steps = starting_epoch * len(self.train_loader)
                elif "step" in training_difference:
                    resume_step = int(training_difference.replace("step_", ""))
                    completed_steps = resume_step
                    starting_epoch = resume_step // len(self.train_loader)
                    resume_step -= starting_epoch * len(self.train_loader)

                # update progress bar if resumed from checkpoint
                if self.accelerator.is_main_process:
                    self.pbar.update(completed_steps)
                self.train_state["step"] = completed_steps
                self.train_state["epoch"] = starting_epoch
            else:
                resume = False
        else:
            resume = False
            
        for epoch_idx in range(starting_epoch, self.num_epoch):
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(epoch_idx)
                
            if resume and len(os.listdir(self.work_dir)) > 0:
                # skip new `skip_first_batches` to skip the batches when resuming from ckpt
                if epoch_idx == starting_epoch and resume_step is not None:
                    # We need to skip steps until we reach the resumed step
                    active_dataloader = self.accelerator.skip_first_batches(self.train_loader, resume_step)
                else:
                    active_dataloader = self.train_loader
            else:
                # After the first iteration though, we need to go back to the original dataloader
                active_dataloader = self.train_loader
                
            self.train_one_epoch(active_dataloader)
            
        if self.accelerator.is_main_process:
            self.pbar.close()
    
    def maybe_save(self):
        if self.save_strategy == "step":
            current_value = self.train_state[self.save_strategy]
            interval = getattr(self, f"save_{self.save_strategy}")
            
            if current_value != 0 and current_value % interval == 0:
                prefix = f"{self.save_strategy}_{int(current_value)}"
                logger.info(f"Saving ckpt to {os.path.join(self.work_dir, prefix)}")
                self.accelerator.save_state(os.path.join(self.work_dir, prefix))
        elif self.save_strategy == "epoch":
            if self.epoch_end:
                prefix = f"{self.save_strategy}_{int(round(self.train_state['epoch'], 1))}"
                logger.info(f"Saving ckpt to {os.path.join(self.work_dir, prefix)}")
                self.accelerator.save_state(os.path.join(self.work_dir, prefix))
                
        if self.save_total_limit is not None:
            saved_ckpts = [f for f in os.listdir(self.work_dir) if "step" in f or "epoch" in f]
            if len(saved_ckpts) > 0 and len(saved_ckpts) > self.save_total_limit:
                saved_ckpts.sort(key=lambda x: int(x.split("_")[1]))
                if self.accelerator.is_main_process:
                    shutil.rmtree(os.path.join(self.work_dir, saved_ckpts[0]))
                    
                
    def maybe_eval(self):
        if self.eval_strategy == "step":
            assert self.eval_step is not None, "Cannot eval by steps without eval_steps"
            current_value = self.train_state[self.eval_strategy]
            interval = getattr(self, f"eval_{self.eval_strategy}")
            
            if current_value != 0 and current_value % interval == 0:
                self.eval_loop()
        elif self.eval_strategy == "epoch":
            if self.epoch_end:
                self.eval_loop()
            
    def log(self, items: dict, prefix: str, to_online_logger: bool = True):
        for k, v in items.items():
            if isinstance(v, torch.Tensor):
                items[k] = v.item()
                
        self.should_log.update(items)
        if self.online_logger is not None and to_online_logger:
            self.online_logger.log(items, self.train_state["step"], prefix)
    

    def display_log(self, pbar):
        now = datetime.now()
        logs = "[" + now.strftime("\033[34m%Y-%m-%d %H:%M:%S\033[0m") + "]"
        logs += str(self.should_log)
        pbar.write(logs)
        