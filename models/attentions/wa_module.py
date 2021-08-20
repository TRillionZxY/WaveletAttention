import torch
import torch.nn as nn
from DWT.DWT_layer import *


class wa_module(nn.Module):
    def __init__(self, wavename='haar'):
        super(wa_module, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    @staticmethod
    def get_module_name():
        return "wa"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)
        output = torch.add(LH, HL)
        output = nn.Softmax(output)
        output = LL + torch.mul(LL, output)
        return output, LL
