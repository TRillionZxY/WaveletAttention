"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''
import functools
import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=True, attention_module=None):
    layers = []
    input_channel = 3

    if attention_module is not None:
        if type(attention_module) == functools.partial:
            m_name = attention_module.func.get_module_name()
        else:
            m_name = attention_module.get_module_name()

    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def VGGWrapper(num_class=10, features=None):
    return VGG(num_class=num_class, features=features)

def VGG11_bn(num_class=10, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        features=make_layers(cfg['A'], attention_module=attention_module))

def VGG13_bn(num_class=10, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        features=make_layers(cfg['B'], attention_module=attention_module))

def VGG16_bn(num_class=10, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        features=make_layers(cfg['D'], attention_module=attention_module))

def VGG19_bn(num_class=10, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        features=make_layers(cfg['E'], attention_module=attention_module))