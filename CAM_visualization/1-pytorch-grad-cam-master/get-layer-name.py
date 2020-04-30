import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torch import nn
import os



if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    def get_last_conv_name(net):

        layer_name = None
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
        return layer_name   

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    #model = models.resnet50(pretrained=True)
    model_name='efficientnet-b2'
    model = EfficientNet.from_name(model_name)
    feature=model._fc.in_features
    model._fc=nn.Linear(in_features=feature,out_features=5,bias=True)
    #checkpoint = torch.load('./checkpoint.pth.tar')
    #model=model.load_state_dict(checkpoint['state_dict'])
    name=get_last_conv_name(model)
    print ('print layer name:', name)   #

    
