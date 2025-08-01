import matplotlib.pyplot as plt
import json
import math


def meepfreq2hz(meep_freq: float) -> float:
    """Convert meep frequency to real life frequency in Hz

    Args:
        meep_freq (float): meep frequency

    Returns:
        float: real life frequency in Hz
    """
    return (3e8/1e-6)*meep_freq

def meepfreq2wl(meep_freq: float) -> float:
    """Convert meep frequency to real life wavelength

    Args:
        meep_freq (float): meep frequency

    Returns:
        float: real life wavelength in nm
    """
    
    return (1 / meep_freq)*1e3

def load_json(path: str):
    with open(path, mode="r") as f:
        file = json.load(f)
        f.close()
        
    return file

def save_json(path: str, obj):
    with open(path, mode="w") as f:
        json.dump(obj, f, indent=2)
        f.close()
        
def make_chunk(seq: list, num_chunks: int) -> list[list]:
    """
    Split input sequence into small chunks

    Args:
        seq (list): The sequence to split
        num_chunks (int): number of chunks

    Returns:
        list[list]: list of chunks, num_chunks in total.
    """
    chunk_size = math.ceil(len(seq) / num_chunks)
    return [seq[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
