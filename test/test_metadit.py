"""
Sample new materials from diffusion model
"""
import torch
import os
from torchvision.utils import save_image
from diffusion import create_diffusion
from model.dit import metadit_s
import argparse
from datapipe import FreeFormDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object, InitProcessGroupKwargs
from datetime import timedelta
from loggers import WrappedLogger
from utils import save_json


local_rank = os.getenv("LOCAL_RANK", -1)
logger = WrappedLogger(__name__)


def main(args):
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), "DDP sample requires GPU"

    diffusion = create_diffusion("", learn_sigma=False)
    model = metadit_s(diffusion=diffusion, condition_channel=602)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=False)
    
    test_set = FreeFormDataset(args.data_path)
    logger.info(f"Test set loaded, total sample {len(test_set)}", on_rank0=True)
    loader = DataLoader(
        test_set, 
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    all_batches = [batch for batch in loader]
    logger.info(f"Batch splitted, total steps {len(all_batches)} with batch size {args.batch_size}", on_rank0=True)
    
    with accelerator.split_between_processes(all_batches, False) as batch_on_rank:
        accelerator.wait_for_everyone()
        pbar = tqdm(total=len(batch_on_rank), desc=f"[rank{local_rank}]")
        all_results = []
        for i, batch in enumerate(batch_on_rank):
            condition = batch["condition"].to("cuda")
            # condition = torch.cat([condition[:, 0, :], condition[:, 1, :]], dim=1)
            z = torch.randn(condition.shape[0], 3, args.resolution, args.resolution, device="cuda")
            z = torch.cat([z, z], 0)  # half for classifier, half for classifier free
            null_condition = torch.ones_like(condition) * 0.5
            condition = torch.cat([condition, null_condition], 0)
            model_kwargs = dict(y=condition, cfg_scale=args.cfg_scale)
            samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device="cuda")
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            
            # remove normalization
            # samples[:, 0] *= 5
            # samples[:, 2] *= 3
            
            for j, item in enumerate(samples):
                data = {
                    "condition": batch["condition"][j].tolist(),
                    "reference": batch["inputs"][j].tolist(),
                    "generation": item.round(decimals=2).cpu().tolist()
                }
                all_results.append(data)
            
            pbar.update(1)
            
        # accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
                
        accelerator.wait_for_everyone()
            
        # distributed storage
        file_name = f"metediff-test-rank{local_rank}.json"
        save_json(os.path.join(args.save_path, file_name), all_results)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="ckpt/xx")
    parser.add_argument("--num_sampling_steps", type=int, default=500)
    parser.add_argument("--data_path", type=str, default="split_data/test_set.mat")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="test/")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--high_res_spec", type=bool, default=False)
    args = parser.parse_args()
    main(args)
