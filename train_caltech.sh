tmpFolder="~/data/caltech256"

# Training Baseline from scratch
python caltech256_main.py --dataset caltech256 --dataset_dir ${tmpFolder} --batch_size 256 --arch resnet18