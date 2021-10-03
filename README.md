# Wavelet Attention CNNs

## Abstract
When Convolutional neural networks (CNN) learns visual features, some key feature details maybe lost in propagation of CNN layers. Generally, attention mechanism is a common solution for CNN to improve the ability of capturing such feature details. However, when CNN uses attention mechanism to capture feature details, it affects the propagation efficiency of main feature information to some extent. In this work, we investigate Discrete Wavelet Transform (DWT) in frequency domain, and design a new Wavelet Attention (WA) mechanism to only implement attention in the high-frequency domain. By embedding WA into CNNs, we propose a novel Wavelet-Attentition CNN (WA-CNN) for effectively learning the visual features of images. Specifically, WA-CNN decomposes the feature maps into low-frequency and high-frequency components for storing the main  information,  as  well  as  the  detail  and  noise  information,  respectively. And then, WA-CNN leverages the WA mechanism to capture the detail information in high-frequency domain with different attention factors, while not affecting the propagation efficiency of feature main information in low-frequency domain. Finally, experimental results on CIFAR-10 and CIFAR-100 datasets show that the proposed WA-CNN achieves the competitive performance in terms of image classification task, which illustrates effectiveness of WA-CNN for learning visual features of images.

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

## Experiments

## Contact
If you have any suggestion or question, you can contact us by: <iezhaoxy@163.com>.  
Thanks for your attention!
