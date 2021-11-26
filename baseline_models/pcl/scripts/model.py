"""model.py"""
import sys
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import json
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))
from model.nn import MLP as MLP_ilcm


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class PCL(nn.Module):
    def __init__(self, z_dim=10, nc=3, architecture="standard_conv", image_shape=None):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.architecture = architecture
        self.image_shape = image_shape
        if self.architecture == "standard_conv":
            assert self.image_shape == (nc, 64, 64)
            self.encoder = nn.Sequential(
                nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
                nn.ReLU(True),
                View((-1, 256*1*1)),                 # B, 256
                nn.Linear(256, z_dim),             # B, z_dim
            )
            self.weight_init()
        elif self.architecture == "ilcm_tabular":
            assert len(self.image_shape) == 1
            self.encoder = MLP_ilcm(image_shape[0], z_dim, 512, 6, spectral_norm=False, batch_norm=False)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=False):
        distributions = self._encode(x)
        return None, distributions, None

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)