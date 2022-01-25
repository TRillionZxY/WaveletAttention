# Wavelet-Attention CNN for Image Classification

[Paper](https://arxiv.org/abs/2201.09271)

## Abstract
The feature learning methods based on convolutional neural network (CNN) have successfully produced tremendous achievements in image classification tasks. However, the inherent noise and some other factors may weaken the effectiveness of the convolutional feature statistics. In this paper, we investigate Discrete Wavelet Transform (DWT) in the frequency domain and design a new Wavelet-Attention (WA) block to only implement attention in the high-frequency domain. Based on this, we propose a Wavelet-Attention convolutional neural network (WA-CNN) for image classification. Specifically, WA-CNN decomposes the feature maps into low-frequency and high-frequency components for storing the structures of the basic objects, as well as the detailed information and noise, respectively. Then, the WA block is leveraged to capture the detailed information in the high-frequency domain with different attention factors but reserves the basic object structures in the low-frequency domain. Experimental results on CIFAR-10 and CIFAR-100 datasets show that our proposed WA-CNN achieves significant improvements in classification accuracy compared to other related networks. Specifically, based on MobileNetV2 backbones, WA-CNN achieves 1.26% Top-1 accuracy improvement on the CIFAR-10 benchmark and 1.54% Top-1 accuracy improvement on the CIFAR-100 benchmark.

## Our environments and toolkits
* OS: Ubuntu 18.04.5  
* Python: 3.8  
* Toolkit: PyTorch 1.8.0  
* CUDA: 11.0  
* GPU: 1080Ti  
* [PyWavelets](https://github.com/PyWavelets/pywt)  
* [torchinfo](https://github.com/TylerYep/torchinfo)  
* [thop](https://github.com/Lyken17/pytorch-OpCounter)(Optional)  

## Usage
### Installtion
```shell
#pip
pip install PyWavelets
pip install torchinfo

#conda
conda install -c conda-forge pywavelets
conda install -c conda-forge torchinfo
```
### Training
```shell
bash train.sh
```

## Contact
If you have any suggestion or question, you can contact us by: <iezhaoxy@163.com>.  
Thanks for your attention!
