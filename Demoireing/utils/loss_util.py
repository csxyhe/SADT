import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *
from torchvision import models as tv
from torch.nn.parameter import Parameter
import os
import scipy.stats as st



class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt1)
        loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam*F.l1_loss(out2, gt2)
        loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam*F.l1_loss(out3, gt3)
        
        return loss1+loss2+loss3            

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class DilatedAdvancedSobelLayer(nn.Module):
    def __init__(self, dilation=1):
        super(DilatedAdvancedSobelLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_rtToLb = [[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]]
        kernel_ltToRb = [[2, 1, 0],
                        [1, 0, -1],
                        [0, -1, -2]]

        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        kernel_rtToLb = torch.FloatTensor(kernel_rtToLb).unsqueeze(0).unsqueeze(0)
        kernel_ltToRb = torch.FloatTensor(kernel_ltToRb).unsqueeze(0).unsqueeze(0)
        self.weight_h = Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = Parameter(data=kernel_v, requires_grad=False)
        self.weight_rtToLb = Parameter(data=kernel_rtToLb, requires_grad=False)
        self.weight_ltToRb = Parameter(data=kernel_ltToRb, requires_grad=False)

        self.dilation = dilation

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # 转换成灰度图
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=self.dilation, dilation=self.dilation)
        x_h = F.conv2d(x, self.weight_h, padding=self.dilation, dilation=self.dilation)
        x_rtToLb = F.conv2d(x, self.weight_rtToLb, padding=self.dilation, dilation=self.dilation)
        x_ltToRb = F.conv2d(x, self.weight_ltToRb, padding=self.dilation, dilation=self.dilation)
        # x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + torch.pow(x_rtToLb, 2) + torch.pow(x_ltToRb, 2) + 1e-6)

        # 返回四个filter对应的输出
        return x_v, x_h, x_rtToLb, x_ltToRb

class DASLoss(nn.Module):
    def __init__(self, dilation=1):
        super(DASLoss, self).__init__()
        self.dilation = dilation
        self.grad_layer = DilatedAdvancedSobelLayer(dilation)
        self.loss = nn.L1Loss()

    def forward(self, out, gt):
        m1, m2, m3, m4 = self.grad_layer(out)
        n1, n2, n3, n4 = self.grad_layer(gt)
        return self.loss(m1, n1) + self.loss(m2, n2) + self.loss(m3, n3) + self.loss(m4, n4)

class MDASLLoss(nn.Module):
    def __init__(self):
        '''
            dilation ={1, 2, 3}
        '''
        super(MDASLLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.dasl1 = DASLoss(dilation=1)
        self.dasl2 = DASLoss(dilation=2)
        self.dasl3 = DASLoss(dilation=3)

    def forward(self, output, gt_img):
        return self.dasl1(output, gt_img) + self.dasl2(output, gt_img) + self.dasl3(output, gt_img)




def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2) # 复制到和输入图片的channels数一致
    return out_filter # (K, K, 3, 1)


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc) # depth-wise 
        return x


class ColorLoss(nn.Module):
    def __init__(self, nc=3):
        super(ColorLoss, self).__init__()
        self.blur = Blur(nc)

    def forward(self, x1, x2):
        x1 = self.blur(x1)
        x2 = self.blur(x2)
        return torch.mean(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class Multi_CharColorDASL_Loss(nn.Module):
    def __init__(self, nc=3):
        '''
            混合了Charbonnier Loss, Dilated Advanced Sobel Loss (\lam_asl), Color Loss (\lam_cr)
        '''
        super(Multi_CharColorDASL_Loss, self).__init__()
        self.lam_asl = 0.3
        self.lam_cr = 0.18
        self.charl = CharbonnierLoss()
        self.mdasl = MDASLLoss()
        self.crl = ColorLoss(nc)

    def forward(self, out1, out2, out3, gt1):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        loss1 = self.charl(out1, gt1) + self.lam_asl*self.mdasl(out1, gt1) + self.lam_cr*self.crl(out1, gt1)
        loss2 = self.charl(out2, gt2) + self.lam_asl*self.mdasl(out2, gt2) + self.lam_cr*self.crl(out2, gt2)
        loss3 = self.charl(out3, gt3) + self.lam_asl*self.mdasl(out3, gt3) + self.lam_cr*self.crl(out3, gt3)

        return loss1 + loss2 + loss3

# if __name__ == '__main__':

#     import cv2

#     asl = Multi_CharColorDASL_Loss()
#     out1 = cv2.imread('/home/xuyi/data/moire/FHDMi/train/source/src_00001.png')

#     a = out1.shape # (256, 256, 3)
    
#     out1 = (out1 / 255.0).astype(np.float32)
#     out1 = torch.from_numpy(out1).permute(2, 0, 1).unsqueeze(0)
#     out2 = F.interpolate(out1, scale_factor=0.5, mode='bilinear', align_corners=False)
#     out3 = F.interpolate(out1, scale_factor=0.25, mode='bilinear', align_corners=False)
#     gt = cv2.imread('/home/xuyi/data/moire/FHDMi/train/target/tar_00001.png')
#     gt = (gt / 255.0).astype(np.float32)
#     gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)
#     res = asl.forward(out1, out2, out3, gt)
#     # res1 = F.l1_loss(img_rgb1, img_rgb4) + 
#     print(res)
#     # print(res1)
