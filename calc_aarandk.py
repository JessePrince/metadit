import os
import torch

from utils import load_json
from model.surrogate import surrogate_s3
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from loggers import WrappedLogger
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple

logger = WrappedLogger(__name__)


def load_multiple_data(path):
    files = os.listdir(path)
    data = [raw for file in files for raw in load_json(os.path.join(path, file))]
    
    return data

def cond_key(arr: np.ndarray) -> bytes:
    """Hash a 1-D array by raw bytes – O(1), zero-copy."""
    return arr.view(np.uint8).tobytes()

# ---------- 2.  Index one list --------------------------------------------
def build_index(recs: List[Dict[str, Any]]) -> Dict[bytes, Dict[str, Any]]:
    """
    { hashed_condition : full_record_dict }
    Keeps the whole record so we can later pull out condition & reference.
    """
    return {cond_key(np.asarray(r["condition"])): r for r in recs}

# ---------- 3.  Align any number of lists ---------------------------------
def align_records(*lists: List[Dict[str, Any]]
                 ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[Any, ...]]]:
    """
    Returns a list of triples:
        (condition, reference, (gen₁, gen₂, …, genₖ))
    One triple for every condition common to *all* input lists.
    """
    indexed = [build_index(lst) for lst in lists]
    common_keys = set.intersection(*(set(ix.keys()) for ix in indexed))

    aligned = []
    for k in common_keys:
        base = indexed[0][k]                          # canonical record
        cond = base["condition"]
        ref  = base["reference"]
        gens = tuple(ix[k]["generation"] for ix in indexed)
        #   ↓ Optional sanity-check that references are truly identical ↓
        # assert all(np.array_equal(ref, ix[k]["reference"]) for ix in indexed)
        aligned.append((cond, ref, gens))
    return aligned

def get_condition(data):
    return [data[0][0], data[0][1]]

def get_reference(data):
    return data[1][0]

def get_generations(data):
    return data[2]

def recover(gen_structure):
    mask = gen_structure[0] < torch.mean(gen_structure[0])
    gen_structure[0][mask] = 0
    gen_structure[0][~mask] = torch.max(gen_structure[0]).clip(0, 1).round(decimals=2)
    mask = gen_structure[1] < torch.mean(gen_structure[1])
    gen_structure[1][mask] = 0
    gen_structure[1][~mask] = torch.max(gen_structure[1]).clip(0, 1).round(decimals=2)
    
    # restore to the original size
    full_structure = torch.zeros(3, 2*gen_structure.shape[1], 2*gen_structure.shape[2], device="cuda")
    left_right = torch.cat([gen_structure[0], gen_structure[0].fliplr()], dim=1)
    full_structure[0] = torch.cat([left_right, left_right.flipud()], dim=0)
    
    left_right = torch.cat([gen_structure[1], gen_structure[1].fliplr()], dim=1)
    full_structure[1] = torch.cat([left_right, left_right.flipud()], dim=0)
    
    full_structure[2] = torch.max(gen_structure[2]).clip(0, 1).round(decimals=2)
    
    return full_structure

def main(args):
    files = os.listdir(args.data_path)
    print(f"Found {len(files)}, Calculating AAE&{len(files)}")
    rollouts = []
    for file in files:
        logger.info(f"Reading from {file}")
        rollouts.append(load_multiple_data(os.path.join(args.data_path, file)))
        
    aligned = align_records(*rollouts)
        
    assert torch.cuda.is_available()
    model = surrogate_s3()
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt, strict=True)
    model.to("cuda")
    model.eval()
    total_samples = len(rollouts[0])
    total_aaeandk = 0
    total_maeandk = 0
    pbar = tqdm(total=total_samples)
    with torch.no_grad():
        for index in range(total_samples):
            gt = torch.tensor(get_condition(aligned[index]), device="cuda", dtype=torch.float32)
            gen_structures = torch.tensor(get_generations(aligned[index]), device="cuda", dtype=torch.float32)
            # binarization
            recovered_structures = [recover(raw) for raw in gen_structures]
            predicts = [model(inputs=struct.unsqueeze(0)).prediction for struct in recovered_structures]
            maes = [torch.mean(torch.abs(predicted_spec - gt)).item() for predicted_spec in predicts]
            aaes = [torch.sum(torch.abs(predicted_spec - gt)).item() for predicted_spec in predicts]
            pbar.write(f"AAEs {aaes} MAEs {maes}")
            total_aaeandk += np.max(aaes)
            total_maeandk += np.max(maes)
            pbar.update(1)
           
    average_aaeandk = total_aaeandk / total_samples
    average_maeandk = total_maeandk / total_samples
    print(f"Average AAE&{len(files)}:", average_aaeandk.item())
    print(f"Average MAE&{len(files)}:", average_maeandk.item())
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    # parser.add_argument("")
    args = parser.parse_args()
    main(args)
    