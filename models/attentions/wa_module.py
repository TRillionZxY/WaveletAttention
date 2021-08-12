import torch as nn
from DWT.DWT_layer import *


class wa_module(nn.Module):
    def __init__(self, wavename='haar'):
        super(wa_module, self).__init__()
        self.dwt = DWT_2D_tiny(wavename=wavename)

    def forward(self, input):
        LL = self.dwt(input)
        return LL
