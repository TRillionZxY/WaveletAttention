import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


# BasicBlock in ResNet for CIFAR
class BasicBlock(nn.Module):

    EXPANSION = 1

    def __init__(self, in_channels, out_channels, stride=1, attention_module=None):
        super(BasicBlock, self).__init__()

        self.flag = False

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                self.m_name = attention_module.func.get_module_name()
            else:
                self.m_name = attention_module.get_module_name()

        if stride != 1 and self.m_name == "wa":
            self.flag = True
            self.wa = attention_module()
            self.conv1 = conv3x3(in_channels, out_channels, stride=1)
        else:
            self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels * self.EXPANSION, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels * self.EXPANSION)

        if (self.m_name is not None) and (self.m_name != "wa"):
            self.bn2 = nn.Sequential(self.bn2, attention_module(out_channels * self.EXPANSION))

        self.shortcut = nn.Sequential()
        if stride != 1 and self.m_name == "wa":
            self.shortcut = nn.Sequential(conv1x1(in_channels, out_channels * self.EXPANSION, stride=1),
                    nn.BatchNorm2d(out_channels * self.EXPANSION))
        elif stride != 1 or in_channels != out_channels * self.EXPANSION:
            self.shortcut = nn.Sequential(conv1x1(in_channels, out_channels * self.EXPANSION, stride=stride),
                    nn.BatchNorm2d(out_channels * self.EXPANSION))

    def forward(self, x):
        identity = x
        
        if self.flag:
            out, identity = self.wa(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)

        return self.relu(out)


# Bottlenect in ResNet for CIFAR
class BottleNect(nn.Module):

    EXPANSION = 4

    def __init__(self, in_channels, out_channels, stride=1, attention_module=None):
        super(BottleNect, self).__init__()

        self.flag = False

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                self.m_name = attention_module.func.get_module_name()
            else:
                self.m_name = attention_module.get_module_name()

        if stride != 1 and self.m_name == "wa":
            self.flag = True
            self.wa = attention_module()
            self.conv2 = conv3x3(in_channels, out_channels, stride=1)
        else:
            self.conv2 = conv3x3(in_channels, out_channels, stride=stride)

        self.conv1 = conv1x1(in_channels, out_channels, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels *
                             self.EXPANSION, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.EXPANSION)

        if (self.m_name is not None) and (self.m_name != "wa"):
            self.bn3 = nn.Sequential(self.bn2, attention_module(out_channels * self.EXPANSION))

        self.shortcut = nn.Sequential()
        if stride != 1 and self.m_name == "wa":
            self.shortcut = nn.Sequential(conv1x1(in_channels, out_channels * self.EXPANSION, stride=1),
                    nn.BatchNorm2d(out_channels * self.EXPANSION))
        elif stride != 1 or in_channels != out_channels * self.EXPANSION:
            self.shortcut = nn.Sequential(conv1x1(in_channels, out_channels * self.EXPANSION, stride=stride),
                    nn.BatchNorm2d(out_channels * self.EXPANSION))

    def forward(self, x):
        identity = x
        
        if self.flag:
            out, identity = self.wa(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)

        return self.relu(out)
