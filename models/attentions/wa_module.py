from os import name
from cifar_main import main
import torch
import torch.nn as nn
from DWT import DWT_2D


class wa_module(nn.Module):
    def __init__(self, wavename='haar'):
        super(wa_module, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.F = torch.add()
        self.softmax = nn.Softmax2d()
        self.delta = torch.mul()

    @staticmethod
    def get_module_name():
        return "wa"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)
        output = self.F(LL, self.delta(LL, self.softmax(self.F(LH, HL))))
        return output, LL

if  __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    print(wa_module(input))