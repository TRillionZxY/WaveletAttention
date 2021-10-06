'''
Visualize FeatureMap
REF: https://github.com/cjf8899/FeatureMap_Visualize_Pytorch.git
'''
from PIL import Image
from torchvision import transforms
from models.cifar import create_net
from utils.checkpoint import load_checkpoint
import matplotlib.pyplot as plt
import argparse
import os

class FM_visualize:
    def __init__(self, module_name, layer_index):
        self.hook = module_name[layer_index].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()


parser = argparse.ArgumentParser(description='Feature Map Visualizing')

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

save_path = "./photograph/"
args.img = save_path + args.img

vpath = args.dataset
vpath += "-" + args.arch
vpath += "-" + args.block_type
if args.attention_type.lower() != "none":
    vpath += "-" + args.attention_type
if args.attention_type.lower() != "none":
    vpath += "-param" + str(args.attention_param)

save_path += vpath
if not os.path.isdir(save_path):
    os.makedirs(save_path)

args.resume += vpath
args.resume += "-nfilters16"
args.resume = os.path.join(args.resume, 'model_best_checkpoint.pth.tar')

net = create_net(args)
net.cuda()
net, _, _, _ = load_checkpoint(args, net)
net.eval()

with open(os.path.join(save_path, "module_name.txt"), mode="wt") as f:
    print(net, file=f)

f.close()

#############################
# Enter the model module and number to visualize
visual = FM_visualize(net.layer2, 0)
# Enter the last channel of the model layer to visualize
out_channal = 16
#############################

img = Image.open(args.img)
trans = transforms.ToTensor()
img = trans(img).unsqueeze(0).cuda()

net(img)
activations = visual.features

rows = int(out_channal/4)
columns = 4
fig, axes = plt.subplots(rows, columns, figsize=(40, 40))

for row in range(rows):
    for column in range(columns):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.imshow(activations[0][row*8+column])

plt.savefig(os.path.join(save_path, 'fmv_result.jpg'), dpi=1000)