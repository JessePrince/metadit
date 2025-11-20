"""
@ 2025 MetaDiT project

MAE, AAE and AAE&K calculation, you should place your data like this
|-- inference_results/
|--|-- seed0.json
|--|-- seed7.json
|--|-- ...
"""

import os
import torch
import re
import numpy as np

from utils import load_json, save_json
from model.surrogate import surrogate_s3
from argparse import ArgumentParser
from loggers import WrappedLogger
from tqdm import tqdm

logger = WrappedLogger(__name__)

NUM_LAYERS = 3

def mean_absolute_error(
    y_true: torch.Tensor | list, 
    y_pred: torch.Tensor | list
) -> float:
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        
    return torch.mean(torch.abs(y_pred - y_true)).item()

def accumulate_absolute_error(
    y_true: torch.Tensor | list, 
    y_pred: torch.Tensor | list
) -> float:
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        
    return torch.sum(torch.abs(y_pred - y_true)).item()

def _flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten_list(item))
        else:
            result.append(item)
    return result

def _extract_seed_number(text):
    pattern = r'seed(\d+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if len(matches) > 1:
        raise ValueError(f"Invalid file name: {matches}")
    elif len(matches) == 1:
        return int(matches[0])
    else:
        return None
    
def build_surrogate_model(model_path, device):
    model = surrogate_s3()
    print(f"Loaded surrogate from {model_path}")
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()
    return model
    
def restore_structure(gen_structure: torch.Tensor) -> torch.Tensor:
    for i in range(NUM_LAYERS-1):
        layer = gen_structure[i]
        mask = layer < torch.mean(layer)
        
        target_val = torch.max(layer).clip(0, 1).round(decimals=2)
        gen_structure[i] = torch.where(mask, torch.tensor(0.0, device=layer.device), target_val)

    H, W = gen_structure.shape[1], gen_structure.shape[2]
    full_structure = torch.zeros(3, 2*H, 2*W, device=gen_structure.device)

    for i in range(NUM_LAYERS-1):
        left_right = torch.cat([gen_structure[i], gen_structure[i].fliplr()], dim=1)
        full_structure[i] = torch.cat([left_right, left_right.flipud()], dim=0)

    full_structure[NUM_LAYERS-1] = torch.max(gen_structure[2]).clip(0, 1).round(decimals=2)
    
    return full_structure

def align_by_condition(data_dict: dict[str: list[dict]]):
    # Find all unique conditions across all seeds
    all_conditions = set()
    for seed_data in data_dict.values():
        for item in seed_data:
            # Convert numpy array to tuple for hashability
            condition_tuple = tuple(_flatten_list(item["condition"]))
            all_conditions.add(condition_tuple)
    
    # Create alignment mapping
    aligned_dict = {}
    for seed_name, seed_data in data_dict.values():
        # Create a mapping from condition to the corresponding dict
        condition_map = {}
        for item in seed_data:
            condition_tuple = tuple(_flatten_list(item["condition"]))
            condition_map[condition_tuple] = item
        
        # Build aligned list for this seed
        aligned_list = []
        for condition in sorted(all_conditions):  # sort for consistent order
            if condition in condition_map:
                aligned_list.append(condition_map[condition])
            else:
                raise ValueError(f"The data is not consistent, please check!")
        
        aligned_dict[seed_name] = aligned_list
    
    return aligned_dict

def validate_alignment(data_dict: dict[str: list[dict]]):
    lists = list(data_dict.values())
    expected_length = len(lists[0])
    
    # Check all lists have same length
    if any(len(lst) != expected_length for lst in lists[1:]):
        raise ValueError("Data is not consistent!")
    
    # Compare conditions
    for items in zip(*lists):
        conditions = [tuple(_flatten_list(item["condition"])) for item in items]
        if not all(cond == conditions[0] for cond in conditions[1:]):
            return False
    return True


def eval_loop(data: list[dict], model, pbar, device):
    maes = []
    aaes = []
    for item in data:
        gt = torch.tensor(item["condition"], device=device, dtype=torch.float32)
        gen_structure = torch.tensor(item["generation"], device=device, dtype=torch.float32)
        full_structure = restore_structure(gen_structure)
        
        predicted_spec = model(inputs=full_structure.unsqueeze(0)).prediction
        
        maes.append(mean_absolute_error(gt, predicted_spec))
        aaes.append(accumulate_absolute_error(gt, predicted_spec))
        
        pbar.update(1)
        
    return maes, aaes

def calculate_aaeandk(data_dict: dict[str: dict], k: int):
    # Extract the first k items and their AAE lists
    aae_lists = [data['aaes'] for data in list(data_dict.values())[:k]]
    
    if not aae_lists:
        return 0.0
    
    transposed_values = list(zip(*aae_lists))
    
    # Calculate maximum at each index and then the overall mean
    maximum_aae = [max(values) for values in transposed_values]
    
    return np.mean(maximum_aae).item()

def main(args):
    torch.set_grad_enabled(False)
    print(f"Evaluate for AAE&{args.k}")
    files = os.listdir(args.data_path)
    data_dict = {}
    for file in files:
        print(f"Loading {file}")
        current_seed = _extract_seed_number(file)
        data_dict[f"seed{current_seed}"] = load_json(
            os.path.join(args.data_path, file)
        )
    if args.k is not None:
        max_k = max(args.k)
        if len(data_dict.keys()) != max_k:
            raise ValueError(
                f"Required to calculate AAE&{max_k} but got only {len(data_dict.keys())} data"
            )
        
    assert "seed0" in data_dict, f"Seed 0 results not found! {data_dict.keys()}"
    total_data_length = sum([len(v) for v in data_dict.values()])
    logger.info(f"Data loaded, total length {total_data_length}")
    
    if len(data_dict) > 1:
        logger.info(f"Validating data alignment...")
        is_aligned = validate_alignment(data_dict)
        if not is_aligned:
            logger.warning("Data is not aligned, perform alignment automatically")
            data_dict = align_by_condition(data_dict)
    
    model = build_surrogate_model(args.model_path, args.device)
    pbar = tqdm(total=total_data_length)
    
    metric_dict = {}
    with torch.no_grad():
        for seed, data in data_dict.items():
            pbar.write(f"Calculating for {seed}")
            maes, aaes = eval_loop(data, model, pbar, args.device)
            metric_dict[seed] = {"maes": maes, "aaes": aaes}
            
    pbar.close()
    final_metric = {
        "MAE": np.mean(metric_dict["seed0"]["maes"]).item(),
        "AAE": np.mean(metric_dict["seed0"]["aaes"]).item()
    }
    
    if args.k is not None:
        for k in args.k:
            # For exact reproducibility, uncomment this, and comment the _data_dict=data_dict below
            # if k == 2:
            #     _metric_dict = {"seed0": metric_dict["seed0"], "seed7": metric_dict["seed7"]}
            # else:
            #     _metric_dict = metric_dict
            _metric_dict = metric_dict
            final_metric[f"AAE&{k}"] = calculate_aaeandk(_metric_dict, k)
        
    print("------ Metrics ------")
    for k, v in final_metric.items():
        print(f"{k}: {v}")
        
    print("---------------------")
    
    if args.metric_save_path:
        parent = os.path.dirname(args.metric_save_path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        save_json(args.metric_save_path, final_metric)

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate metrics for MetaDiT")
    # Path
    parser.add_argument(
        "--data_path", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--metric_save_path",
        type=str,
        default=None
    )
    
    # Model
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    
    # Metric settings
    parser.add_argument(
        "--k",
        type=int,
        nargs='+',
        help="Input multiple K value to specify multiple AAE&K calculation"
    )
    
    args = parser.parse_args()
    main(args)
