#pragma once
#include <torch/extension.h>
#include <ATen/div_rtn.h>
at::Tensor
dcn_v3_cuda_forward(const at::Tensor &input,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int deformable_group);

std::vector<at::Tensor>
dcn_v3_cuda_backward(const at::Tensor &input,
                     const at::Tensor &offset,
                     const at::Tensor &mask,
                     const at::Tensor &grad_output,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int pad_h, int pad_w,
                     int dilation_h, int dilation_w,
                     int deformable_group);

