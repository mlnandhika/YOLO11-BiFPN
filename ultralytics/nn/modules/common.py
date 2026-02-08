import torch.nn as nn

class GetItem(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]

class Pack(nn.Module):
    def forward(self, x):
        return x