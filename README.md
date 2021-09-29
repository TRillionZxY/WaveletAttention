# Wavelet Attention CNNs

## Our environments and toolkits
* OS: Ubuntu 18.04.5  
* Python: 3.8  
* Toolkit: PyTorch 1.8.0  
* CUDA: 11.0  
* GPU: 1080Ti  
* [PyWavelets](https://github.com/PyWavelets/pywt)  
* [torchinfo](https://github.com/TylerYep/torchinfo)  
* [thop](https://github.com/Lyken17/pytorch-OpCounter)(Optional)  


## Abstract
When Convolutional neural networks (CNNs) learns visual features, the feature details are lost to a certain extent in propagation. Attention mechanism is a common solution for CNNs to improve the ability of capturing feature details. However, while CNNs use attention mechanism to capture detailed feature information, it affects the transmission of feature main information, but not affects the destruction of features by down-sampling operation. The Non-Local Block presents a pioneering approach for capturing the global context features, while it has high training costs. The Non-Local Block can be abstracted as a general framework for global context modeling. Within the general framework, we design a new attention mechanism based on Discrete Wavelet Transform (DWT), which is the Wavelet Attention Block. In our method, feature maps are decomposed into low-frequency and high-frequency components through DWT during the down-sampling. The low-frequency components retain the main information of feature maps, and the high-frequency components contain the details and noise information. Then, our method captures the global context features through the detail information in high-frequency components, which does not affect the main information transmission in the low-frequency components. Moreover, we propose the Wavelet Attention CNNs, which enhance the feature learning ability of CNNs for image classification. Our experimental results on CIFAR verify the effectiveness of the Wavelet Attention CNNs.
## Experiments

## Contact
If you have any suggestion or question, you can contact us by: <iezhaoxy@163.com>.  
Thanks for your attention!
