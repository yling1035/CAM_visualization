from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
import torch as t
import os
from torchvision import models,datasets,transforms
from PIL import Image
import copy

def preprocess_image(img):
    """
    预处理层
    将图像进行标准化处理
    """
    mean = [0.485, 0.456, 0.406] 
    stds = [0.229, 0.224, 0.225]
    #preprocessed_img = img.copy()[:, :, ::-1] # BGR > RGB
    preprocessed_img = copy.copy(img)
    
    #标准化处理， 将bgr三层都处理
    for i in range(3):

        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1))) #transpose HWC > CHW
    preprocessed_img = t.from_numpy(preprocessed_img) #totensor
    preprocessed_img.unsqueeze_(0)
    input = t.tensor(preprocessed_img, requires_grad=True)
    
    return input

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


model = EfficientNet.from_name('efficientnet-b0')
img = cv2.imread('./images/feed.png') #读取图像
# = Image.img_to_array(img)
x=t.from_numpy(img)
x=np.float32(x)/255
#x = np.expand_dims(x, axis=0)
print('print x:',x)
#x = preprocess_input(x)
x=preprocess_image(x)
print('Input image shape:', x)
'''
img = np.float32(cv2.resize(img, (260,260))) / 255
input = preprocess_image(img)
'''
#rint(img.shape) # torch.Size([1, 3, 224, 224])
features = model.extract_features(x)
print(features.shape) # torch.Size([1, 1280, 7, 7])