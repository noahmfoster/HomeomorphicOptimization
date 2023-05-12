import torch
from torch.utils.data import Dataset

class CorrelatedDataset(Dataset):
    def __init__(self, num_samples=1000):
        super().__init__()
        self.num_samples = num_samples
        
        # Generate random data with correlations
        self.data = torch.randn(num_samples, 10)
        self.data[:, :5] += 2.0
        self.data[:, 5:] -= 2.0
        self.labels = torch.zeros(num_samples, dtype=torch.long)
        self.labels[:num_samples//2] = 1
        
        idx = torch.randperm(num_samples)
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
