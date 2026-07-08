# model/encoder.py
import torch.nn as nn
from config import *

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=6,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=ENCODER_LAYERS,
        dim_feedforward=ENCODER_FF_HIDDEN,
        dropout=ENCODER_DROPOUT,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        return self.encoder(self.proj(x))
