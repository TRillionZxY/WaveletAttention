"""
VGG in PyTorch
REF: Very Deep Convolutional Networks for Large-Scale Image Recognition.
"""
import functools
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=10, init_weights=True):
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

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

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

        if m_name == "wad":
            for l in cfg:
                if l == 'M':
                    layers += [attention_module()]
                    continue

                layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

                if batch_norm:
                    layers += [nn.BatchNorm2d(l)]

                layers += [nn.ReLU(inplace=True)]
                input_channel = l
        else:
            for l in cfg:
                if l == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    continue

                layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

                if batch_norm:
                    layers += [nn.BatchNorm2d(l), attention_module(l)]

                layers += [nn.ReLU(inplace=True)]
                input_channel = l
    else:
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

def VGGWrapper(num_class=10, features=None, block=None):
    return VGG(num_class=num_class, features=features)

def VGG11_bn(num_class=10, block=None, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        block=block,
        features=make_layers(cfg['A'], attention_module=attention_module))

def VGG13_bn(num_class=10, block=None, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        block=block,
        features=make_layers(cfg['B'], attention_module=attention_module))

def VGG16_bn(num_class=10, block=None, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        block=block,
        features=make_layers(cfg['D'], attention_module=attention_module))

def VGG19_bn(num_class=10, block=None, attention_module=None):
    return VGGWrapper(
        num_class=num_class,
        block=block,
        features=make_layers(cfg['E'], attention_module=attention_module))