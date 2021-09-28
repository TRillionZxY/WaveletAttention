import torch
import torch.nn as nn
from DWT import DWT_2D


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
        output = torch.Tensor(nn.Softmax(output)).cuda()
        output = LL + torch.mul(LL, output)
        return output, LL
