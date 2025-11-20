"""
@ 2025 MetaDiT project

Generate new materials from MetaDiT, supports multi-GPU
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import time

from torch.utils.data import DataLoader
from datetime import timedelta
from loggers import WrappedLogger
from argparse import ArgumentParser
from utils import save_json
from model.dit import DIT_MODEL, DiT
from diffusion import create_diffusion, SpacedDiffusion
from datapipe import FreeFormDataset
from tqdm import tqdm

logger = WrappedLogger(__name__)


def setup_distributed(rank: int, world_size: int):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # An unused port
    # Initialize the process group
    # To avoid NCCL timeout
    TIMEOUT = timedelta(hours=24)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, 
        device_id=torch.device(f"cuda:{rank}"),
        timeout=TIMEOUT
    )
    # Pin the current process to a specific GPU
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()
    torch.cuda.empty_cache()

def split_data(data: list[dict], world_size: int):
    """
    Splits a list of data into 'world_size' chunks, handling uneven splits.
    Uses numpy.array_split for a simple and robust split.
    """
    # np.array_split will create as-equal-as-possible chunks.
    # It returns a list of numpy arrays, so we convert them back to lists.
    return [arr.tolist() for arr in np.array_split(data, world_size)]

def build_model(
    model_path: str, 
    model_type: str, 
    time_steps: int,
    condition_channel: int
) -> tuple[DiT, SpacedDiffusion]:
    diffusion = create_diffusion(str(time_steps), learn_sigma=False)
    model = DIT_MODEL[model_type](
        diffusion=diffusion, condition_channel=condition_channel
    )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)
    
    print("Model loaded on CPU")
    
    return model, diffusion
    

def prepare_data(data_path: str, batch_size: int) -> list[dict]:
    dataset = FreeFormDataset(data_path)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return [batch for batch in loader]

def generate_one_batch(
    payload: dict,
    sampling_params: dict,
    model: DiT,
    diffusion: SpacedDiffusion,
    device: str
) -> list[dict]:
    resolution = sampling_params["resolution"]
    cfg_scale = sampling_params["cfg_scale"]
    
    condition = payload["condition"].to(device)
    z = torch.randn(
        condition.shape[0], 
        3, 
        resolution, 
        resolution, 
        device=device
    )
    z = torch.cat([z, z], 0)  # half for classifier, half for classifier free
    null_condition = torch.ones_like(condition) * 0.5
    condition = torch.cat([condition, null_condition], 0)
    model_kwargs = dict(y=condition, cfg_scale=cfg_scale)
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, 
        z.shape, 
        z, 
        clip_denoised=False, 
        model_kwargs=model_kwargs, 
        progress=True, 
        device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    
    batch_result = [
        {
            "condition": payload["condition"][j].tolist(),
            "reference": payload["inputs"][j].tolist(),
            "generation": sample.round(decimals=2).cpu().tolist()
        }
        for j, sample in enumerate(samples)
    ]
                
    return batch_result


def single_gpu_inference(
    model: DiT, 
    data: list[dict],
    sampling_params: dict,
    diffusion: SpacedDiffusion,
    gpu_id: int = 0
):
    """
    Runs inference on a single GPU.
    """
    device = f"cuda:{gpu_id}"
    print(f"--- Running single-GPU inference on {device} ---")
    
    model.to(device)
    model.eval()

    all_results = []
    
    def generate_and_store(payload: dict, storage: list):
        batch_result = generate_one_batch(
            payload, sampling_params, model, 
            diffusion, device
        )
        storage.extend(batch_result)
    
    with torch.no_grad():
        for item in tqdm(data, desc="Single-GPU Inference"):
            generate_and_store(item, all_results)

    return all_results

def worker_fn(
    rank: int, 
    world_size, 
    data_path,
    model_path, 
    model_type,
    time_steps,
    condition_channel,
    sampling_params,
    temp_dir,
    seed=0
):
    """
    The main function for each parallel worker process.
    """
    print(f"[Rank {rank}] Worker process started.")
    setup_distributed(rank, world_size)
    
    torch.manual_seed(seed)
    
    model, diffusion = build_model(
        model_path, model_type, time_steps, condition_channel
    )
    
    device = f"cuda:{rank}"
    model.to(device)
    model.eval()

    # Get this rank's chunk of data
    data = prepare_data(data_path, sampling_params["batch_size"])
    data_chunks = split_data(data, world_size)
    local_data = data_chunks[rank]
    
    print(f"[Rank {rank}] local data size {len(local_data)}")
    dist.barrier()
    
    local_results = []
    
    iterable = tqdm(
        local_data, 
        desc=f"Rank {rank} Inference", 
        position=rank,
        leave=True
    )
    
    def generate_and_store(payload: dict, storage: list):
        batch_result = generate_one_batch(
            payload, sampling_params, model, 
            diffusion, device
        )
        storage.extend(batch_result)
    
    with torch.no_grad():
        for item in iterable:
            generate_and_store(item, local_results)
            
    iterable.close()

    # Save this rank's results to a temporary file
    temp_file = os.path.join(temp_dir, f"results_rank_{rank}.pt")
    torch.save(local_results, temp_file)
    print(f"[Rank {rank}] Results saved to {temp_file}")

    dist.barrier()
    cleanup_distributed()
    print(f"[Rank {rank}] Worker process finished.")

def main(args):
    assert torch.cuda.is_available(), "Inference supports GPU only!"
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise ValueError("Error: No GPUs available.")

    if args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs, but only {available_gpus} are available.")
        args.num_gpus = available_gpus
        
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    
    model, diffusion = None, None
    if args.num_gpus == 1:
        print("Loading model onto CPU...")
        model, diffusion = build_model(
            args.model_path, args.model_type, args.condition_channel
        )
        data = prepare_data(args.data_path, args.batch_size)
        print(f"Loaded data size {len(data)}")
    # In multi-GPU inference, we create model, dataset inside the worker,
    # to avoid shm overflow

    all_results = []
    
    sampling_params = {
        "resolution": args.resolution,
        "cfg_scale": args.cfg_scale,
        "batch_size": args.batch_size
    }
    
    start_time = time.time()
    if args.num_gpus == 1:
        all_results = single_gpu_inference(
            model, data, sampling_params, diffusion, gpu_id=0
        )
    else:
        print(f"--- Spawning {args.num_gpus} processes for parallel inference ---")
        world_size = args.num_gpus
        os.makedirs(args.temp_dir, exist_ok=True)
        
        worker_args = (
            world_size, args.data_path, args.model_path, args.model_type,
            args.time_steps, args.condition_channel, sampling_params, 
            args.temp_dir, args.seed 
        )
        mp.spawn(
            worker_fn,
            args=worker_args,
            nprocs=world_size,
            join=True
        )
        
        print("All workers finished. Gathering results...")
        all_results_split = []
        for i in range(world_size):
            temp_file = os.path.join(args.temp_dir, f"results_rank_{i}.pt")
            try:
                chunk = torch.load(temp_file)
                all_results_split.append(chunk)
                os.remove(temp_file) # Clean up
            except FileNotFoundError:
                print(f"Error: Could not find temp file {temp_file}")
        
        all_results = [item for sublist in all_results_split for item in sublist]
        
        # Clean up temp directory
        try:
            os.rmdir(args.temp_dir)
        except OSError:
            pass # Directory might not be empty, which is fine.

    end_time = time.time()
    
    # --- Done ---
    print("\n--- Inference Complete ---")
    print(f"Total results gathered: {len(all_results)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    parent = os.path.dirname(args.save_path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    save_json(all_results, args.save_path)
    
    exit(0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference script for MetaDiT")
    # Infra settings
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for inference. Defaults to 1."
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="cache/inference"
    )
    
    # data information
    parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )
    
    # Model information
    parser.add_argument(
        "--model_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="metadit_s"
    )
    parser.add_argument(
        "--condition_channel",
        type=int,
        default=301
    )
    
    # Sampling Params
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=500
    )
    
    # Save
    parser.add_argument(
        "--save_path",
        type=str,
        default="test/results.json"
    )
    
    args = parser.parse_args()
    
    main(args)
