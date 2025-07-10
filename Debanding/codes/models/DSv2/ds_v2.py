#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import _ext2 as _backend


class _DSMv3(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, kernel_size,
                stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(kernel_size)
        ctx.deformable_groups = deformable_groups
        # print("x.dtype is {}".format(input.dtype))
        # print("mask.dtype is {}".format(mask.dtype))
        # print("offset.dtype is {}".format(offset.dtype))
        output = _backend.dcn_v3_forward(input,
                                         offset, 
                                         mask,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors
        grad_input, grad_offset, grad_mask = \
            _backend.dcn_v3_backward(input,
                                     offset,
                                     mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.deformable_groups)
        # 反向传播函数的返回值应该与前向传播函数的输入值数量保持一致，并且顺序也应该保持一致
        return grad_input, grad_offset, grad_mask, None, None, None, None, None,


dsm_v3 = _DSMv3.apply


class DSMv3(nn.Module):

    def __init__(self, in_channels, 
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DSMv3, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups


    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            mask.shape[1]
        return dsm_v3(input, offset, mask,
                           self.kernel_size,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)


class DSM(DSMv3):
    def __init__(self, in_channels, 
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DSM, self).__init__(in_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)
        self.deformable_groups = deformable_groups
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dsm_v3(input, offset, mask,
                           self.kernel_size, 
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)

