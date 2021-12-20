import os
import torch
from torchvision import transforms
from models.cifar import create_net
from utils.checkpoint import load_checkpoint
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Testing')

# Model settings
parser.add_argument("--dataset", type=str, default="cifar10",
                        help="training dataset (default: cifar10)")
parser.add_argument("--arch", type=str, default="resnet18",
                        help="network architecture (default: resnet18)")
parser.add_argument("--block_type", type=str, default="basic",
                        help="building block for network (possible choices basic|bottlenect|ivrd|vgg")
parser.add_argument("--attention_type", type=str, default="none",
                        help="attention type in building block (possible choices none|se|cbam|eca|gc|wa)")
parser.add_argument("--attention_param", type=str, default="4",
                        help="attention parameter (reduction in (cbam/se/gc), kernel_size in eca(set 3), wavename in wa)")

# Path settings
parser.add_argument('--img', type=str, default="dog.jpg",
                        help="image name")
parser.add_argument("--resume", type=str, default="./ckpts/",
                        help="path to checkpoint for continous training (default: none)")

args = parser.parse_args()

if args.dataset == "cifar10":
    args.num_class = 10
elif args.dataset == "cifar100":
    args.num_class = 100

CFAIR10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck']

normlizer = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normlizer])

save_path = "./photograph/"
args.img = save_path + args.img

vpath = args.dataset
vpath += "-" + args.arch
vpath += "-" + args.block_type
if args.attention_type.lower() != "none":
    vpath += "-" + args.attention_type
    vpath += "-param" + str(args.attention_param)

save_path += vpath
if not os.path.isdir(save_path):
    os.makedirs(save_path)

args.resume += vpath
args.resume += "-nfilters8"
args.resume = os.path.join(args.resume, 'model_best_checkpoint.pth.tar')

net = create_net(args)
net.cuda()
net, _, _, _ = load_checkpoint(args, net)
net.eval()

img = Image.open(args.img)
img_T = transform(img).unsqueeze(0).cuda()

output = net(img_T)
# 求指定维度的最大值，返回最大值以及索引
predict_value, predict_idx = torch.max(output, 1)  

print('predict_value: {:.3f}'.format(predict_value.data))
print(CFAIR10_names[predict_idx])

plt.figure()
plt.imshow(np.array(img))
plt.title(CFAIR10_names[predict_idx])
plt.axis('off')
plt.savefig(os.path.join(save_path, 'predict_result.jpg'))
    