tmpFolder="~/data/caltech256"

# Training Baseline from scratch
python caltech256_main.py --dataset caltech256 --dataset_dir ${tmpFolder} -b 256 --arch resnet18