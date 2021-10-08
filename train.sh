tmpFolder = "~/data"

# Training Baseline from scratch
python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --batch_size 256 --block_type basic --arch resnet18

# Training Attention from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --batch_size 256 --block_type basic --arch resnet18 --attention_type cbam

# Training ResNet-WA from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --batch_size 256 --block_type basic --arch resnet18 --attention_type wa --attention_param haar

# Training VGG-WA from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --batch_size 256 --block_type vgg --arch vgg16_bn --attention_type wad --attention_param haar

# Training MobileNetV2-WA from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --batch_size 256 --block_type ivrd --arch mobilenetv2 --attention_type wad --attention_param haar