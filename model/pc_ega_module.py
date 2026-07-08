# model/pc_ega_module.py
import torch.nn as nn
from config import *

class FiLM(nn.Module):
    def __init__(self, pref_dim=2, d_model=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pref_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)
        )
    def forward(self, pref):
        gamma, beta = self.net(pref).chunk(2, dim=-1)
        return gamma, beta