import torch
import cv2
import numpy as np
from DWT.DWT import DWTFunction_2D, IDWTFunction_2D
from DWT import DWT_2D, IDWT_2D
from datetime import datetime
from torch.autograd import gradcheck


image_full_name = '/Users/ZHAO/data/lena/lena512color.tiff'

image = cv2.imread(image_full_name, flags = 1)
image = image[0:512,0:512,:]
print(image.shape)
t0 = datetime.now()
for index in range(100):
    m0 = DWT_2D(wavename="haar")
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(dim = 0)
    print('image_re shape: {}'.format(image_tensor.size()))
    image_tensor.transpose_(1,3)
    print('image_re shape: {}'.format(image_tensor.size()))
    image_tensor.transpose_(2,3)
    print('image_re shape: {}'.format(image_tensor.size()))
    image_tensor.requires_grad = False
    LL, LH, HL, HH = m0(image_tensor)
    matrix_low_0 = torch.Tensor(m0.matrix_low_0)
    matrix_low_1 = torch.Tensor(m0.matrix_low_1)
    matrix_high_0 = torch.Tensor(m0.matrix_high_0)
    matrix_high_1 = torch.Tensor(m0.matrix_high_1)

    # image_tensor.requires_grad = True
    # input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
    # test = gradcheck(DWTFunction_2D.apply, input)
    # print(test)
    # print(LL.requires_grad)
    # print(LH.requires_grad)
    # print(HL.requires_grad)
    # print(HH.requires_grad)
    # LL.requires_grad = True
    # input = (LL.double(), LH.double(), HL.double(), HH.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
    # test = gradcheck(IDWTFunction_2D.apply, input)
    # print(test)

    m1 = IDWT_2D(wavename="haar")
    image_re = m1(LL,LH,HL,HH)
t1 = datetime.now()
image_re.transpose_(2,3)
image_re.transpose_(1,3)
image_re_np = image_re.detach().numpy()
print('image_re shape: {}'.format(image_re_np.shape))

image_zero = image - image_re_np[0]
print(np.max(image_zero), np.min(image_zero))
print(image_zero[:,8])
print('taking {} secondes'.format(t1 - t0))
cv2.imshow('reconstruction', image_re_np[0]/255)
cv2.imshow('image_zero', image_zero/255)
cv2.waitKey(5000)