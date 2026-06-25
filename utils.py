import numpy as np
import torch
from torch.utils.data import Dataset

class CyberPhysicalTrafficDataset(Dataset):
    """
    Generates synchronized multi-view sequence observations representing 
    instantaneous flow statistics, structural correlations, and finite-difference derivatives.
    """
    def __init__(self, num_samples=200, seq_len=10, base_dim=12, anomaly=False):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        if not anomaly:
            t = np.linspace(0, 50, num_samples * seq_len).reshape(num_samples, seq_len, 1)
            base_data = np.sin(t) * 2.0 + np.random.normal(0, 0.1, (num_samples, seq_len, base_dim))
        else:
            t = np.linspace(0, 50, num_samples * seq_len).reshape(num_samples, seq_len, 1)
            base_data = np.sin(t * 1.5) * 4.0 + np.random.normal(0, 1.5, (num_samples, seq_len, base_dim))

        self.x_inst = torch.tensor(base_data, dtype=torch.float32)
        
        # Structural View: Vectorized covariance window metrics (Section IV)
        structural_mats = []
        for i in range(num_samples):
            window_cov = np.cov(base_data[i].T)
            structural_mats.append(window_cov.flatten()[:base_dim * seq_len])
        
        self.x_struct = torch.tensor(np.array(structural_mats), dtype=torch.float32).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Temporal Derivative View: First-order finite differences (Section IV)
        padded = np.pad(base_data, ((0,0), (1,0), (0,0)), mode='edge')
        derivatives = np.diff(padded, axis=1)
        self.x_deriv = torch.tensor(derivatives, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x_inst[idx], self.x_struct[idx], self.x_deriv[idx]