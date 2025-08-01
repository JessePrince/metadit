"""
Dataset pipeline for model training
"""

import torch.utils

from torch.utils.data.dataset import Dataset
from utils import load_json
import math
import torch.nn.functional as F
from scipy import io


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str
    ):
        super().__init__()
        self.data = io.loadmat(data_path)
        
    def create_3d_grid(self, geometry: dict):
        pass
        
    def __len__(self):
        return len(self.data["parameter"])
    
    def __getitem__(self, index):
        pass
    
    
class FreeFormDataset(BaseDataset):
    def __init__(self, path):
        super().__init__(path)
    
    def create_3d_grid(self, geometry):
        # channel 0 for r_index
        # channel 1 for thickness
        # channel 2 for lattice_size
        grid = torch.zeros(3, 32, 32, dtype=torch.float32)
        # get 1/4 because of symmetry
        pattern = torch.tensor(geometry["pattern"][:32, :32], dtype=torch.float32)
        # normalization
        index = geometry["params"][2] / 5.0
        lattice_size = geometry["params"][0] / 3.0
        
        grid[0][pattern==1] = index
        grid[1][pattern==1] = geometry["params"][1]
        grid[2] = lattice_size
        
        return grid
    
    def __getitem__(self, index):
        # geometry
        geometry = {
            "pattern": self.data["pattern"][:,:,index], # np.array(64, 64)
            "params": self.data["parameter"][index] # [Lattice size, Thicknesses, Refractive index]
        }
        
        """
        metadata: lattice size 2.5 µm to 3 µm
        Thickness: 0.5um to 1um
        refractive index: 3.5 to 5
        """
        
        grid = self.create_3d_grid(geometry)
        # 301 freq points
        imag = torch.tensor(self.data["imag"][index], dtype=torch.float32)
        real = torch.tensor(self.data["real"][index], dtype=torch.float32)
        condition = torch.stack([real, imag], dim=0)
        
        data = {
            "inputs": grid,  # 3, 64, 64
            "condition": condition,  # 2, 301
            "labels": grid
        }
        
        return data
    
    
class SurrogateFreeFormDataset(BaseDataset):
    def __init__(self, path):
        super().__init__(path)
    
    def create_3d_grid(self, geometry):
        # channel 0 for r_index
        # channel 1 for thickness
        # channel 2 for lattice_size
        grid = torch.zeros(3, 64, 64, dtype=torch.float32)
        # get 1/4 because of symmetry
        pattern = torch.tensor(geometry["pattern"], dtype=torch.float32)
        # normalization
        index = geometry["params"][2] / 5.0
        lattice_size = geometry["params"][0] / 3.0
        
        grid[0][pattern==1] = index
        grid[1][pattern==1] = geometry["params"][1]
        grid[2] = lattice_size
        
        return grid
    
    def __getitem__(self, index):
        # geometry
        geometry = {
            "pattern": self.data["pattern"][:,:,index], # np.array(64, 64)
            "params": self.data["parameter"][index] # [Lattice size, Thicknesses, Refractive index]
        }
        
        """
        metadata: lattice size 2.5 µm to 3 µm
        Thickness: 0.5um to 1um
        refractive index: 3.5 to 5
        """
        
        grid = self.create_3d_grid(geometry)
        # 301 freq points
        imag = torch.tensor(self.data["imag"][index], dtype=torch.float32)
        real = torch.tensor(self.data["real"][index], dtype=torch.float32)
        condition = torch.stack([real, imag], dim=0)
        
        data = {
            "inputs": grid,  # 3, 64, 64
            "labels": condition
        }
        
        return data
