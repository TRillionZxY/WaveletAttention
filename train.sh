tmpFolder=""

# Training Baseline from scratch
python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --batch_size 256 --arch resnet18

# # Training Attention from scratch
# python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --arch resnet18 --attention_type cbam

# # Training WA from scratch
# python cifar_main.py --dataset cifar10 --dataset_dir ${tmpFolder} --arch resnet18 --attention_type wa --attention_param haar