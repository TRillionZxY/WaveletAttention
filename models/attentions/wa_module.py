from cifar_main import main
import torch
import torch.nn as nn
from DWT import DWT_2D


class wa_module(nn.Module):
    def __init__(self, wavename='haar'):
        super(wa_module, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wa"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)
        output = LL

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        return output