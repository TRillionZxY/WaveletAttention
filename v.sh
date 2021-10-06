# Visualize Baseline
#python visualize.py --dataset cifar10 --block_type basic --arch resnet18 --img dog.jpg
# Visualize Attention
#python visualize.py --dataset cifar10 --block_type basic --arch resnet18 --attention_type cbam --img dog.jpg
# Visualize ResNet-WA
#python visualize.py --dataset cifar10 --block_type basic --arch resnet18 --attention_type wa --attention_param haar --img dog.jpg
# Visualize VGG-WA
python visualize.py --dataset cifar10 --block_type vgg --arch vgg16_bn --attention_type wad --attention_param haar --img dog.jpg
# Visualize MobileNetV2-WA
#python visualize.py --dataset cifar10 --block_type ivrd --arch mobilenetv2 --attention_type wad --attention_param haar --img dog.jpg