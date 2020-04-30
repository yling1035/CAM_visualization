# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import re
import os
import numpy as np
import torch
from torch import nn
from torchvision import models
import argparse
from skimage import io
import cv2
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
import numpy as np
from torch.autograd import Function
from torchvision import models
from efficientnet_pytorch import EfficientNet
from PIL import Image

def get_net(net_name, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    # 加载指定路径的权重参数
    if weight_path is not None and net_name.startswith('densenet'):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(weight_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
    elif weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./feed.png',
                        help='Input image path')
    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbers | 0,1,2,3 | ")
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def main(args):
    args = get_args()
    # 输入
    img = io.imread(args.image_path)
    #img = Image.open(args.image_path)
    #img=img.convert("RGB")
    img = img[:,:,:3] #将4通道图像转换为3通道图像
    img = np.float32(cv2.resize(img, (260,260))) / 255
    inputs = preprocess_image(img)
    # 输出图像
    image_dict = {}
    # 网络
    model_name='efficientnet-b2'
    model = EfficientNet.from_name(model_name)
    #model=EfficientNet.from_pretrained(model_name)
    feature=model._fc.in_features
    model._fc=nn.Linear(in_features=feature,out_features=5,bias=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch_device = torch.device("cuda")
    model = torch.nn.DataParallel(model).to(torch_device)
    model=model.cuda()
    #print('print models', model)
    #/save_models
    checkpoint=torch.load('./checkpoint.pth.tar')
    net=model.load_state_dict(checkpoint['state_dict'])
    print('print net:',net)
    # Grad-CAM
    layer_name = get_last_conv_name(model) #if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(model, net, layer_name)
    target_index=None
    mask = grad_cam(inputs, target_index)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(model,net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, target_index)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(model, net)
    inputs.grad.zero_()  # 梯度置零
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = gb
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)
    output_dir='./results'
    save_image(image_dict, os.path.basename(args.image_path), model_name, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./feed.png',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
