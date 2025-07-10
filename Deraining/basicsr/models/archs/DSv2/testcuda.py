#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from ds_v2 import dsm_v3, DSMv3, DSM

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3



def example_dconv():
    input = torch.randn(2, 64, 128, 128).cuda()
    conv_mask = nn.Conv2d(64, 3 ** 2, kernel_size=3, padding=1).cuda()
    conv_offset = nn.Conv2d(64, 2 * 3 ** 2, kernel_size=3, padding=1).cuda()
    # wrap all things (offset and mask) in DCN
    offset = conv_offset(input)
    mask = conv_mask(input)
    dsm = DSMv3(64, kernel_size=3, stride=1,
              padding=1).cuda()
    # print(dcn.weight.shape, input.shape)
    output = dsm(input, offset, mask)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)




if __name__ == '__main__':

    example_dconv()

