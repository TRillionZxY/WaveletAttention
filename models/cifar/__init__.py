import functools
from ..attentions import get_attention_module
from .block import BasicBlock, BottleNect, InvertedResidualBlock
from .resnet import ResNet18, ResNet34, ResNet50
from .vgg import VGG16_bn
from .mobilenetv2 import MobileNetV2Wrapper

model_dict = {
    "resnet18": ResNet18, 
    "resnet34": ResNet34, 
    "resnet50": ResNet50,
    "vgg16_bn": VGG16_bn,
    "mobilenetv2": MobileNetV2Wrapper,
}

def get_block(block_type="basic"):

    block_type = block_type.lower()

    if block_type == "basic":
        b = BasicBlock
    elif block_type == "bottlenect":
        b = BottleNect
    elif block_type == "ivrd":
        b = InvertedResidualBlock
    elif block_type == "vgg":
        b = "vgg"
    else:
        raise NotImplementedError(
            'block [%s] is not found for dataset [%s]' % block_type)
    return b


def create_net(args):
    net = None

    block_module = get_block(args.block_type)
    attention_module = get_attention_module(args.attention_type)

    if args.attention_type == "se" or args.attention_type == "cbam":
        attention_module = functools.partial(
            attention_module, reduction=float(args.attention_param))
    elif args.attention_type == "wa":
        attention_module = functools.partial(
            attention_module, wavename=args.attention_param)

    net = model_dict[args.arch.lower()](
        num_class=args.num_class,
        block=block_module,
        attention_module=attention_module
    )

    return net
