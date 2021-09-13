tmpFolder="~/data"

# Training Baseline from scratch
python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --block_type basic --batch_size 256 --arch resnet18

# Training Attention from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --block_type basic --batch_size 256 --arch resnet18 --attention_type cbam

# Training WA from scratch
#python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --block_type basic --batch_size 256 --arch resnet18 --attention_type wa --attention_param haar