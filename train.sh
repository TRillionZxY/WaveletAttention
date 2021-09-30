tmpFolder="~/data"

# Training Baseline from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --block_type basic --batch_size 256 --arch resnet18

# Training Attention from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --block_type basic --batch_size 256 --arch resnet18 --attention_type cbam

# Training ResNet-WA from scratch
python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --block_type basic --batch_size 256 --arch resnet18 --attention_type wa --attention_param haar

# Training VGG-WA from scratch
python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --block_type vgg --batch_size 256 --arch vgg16_bn --attention_type wa --attention_param haar