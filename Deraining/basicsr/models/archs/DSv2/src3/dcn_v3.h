#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor
dcn_v3_forward(const at::Tensor &input,
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
               const int deformable_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v3_cuda_forward(input, offset, mask,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v3_cpu_forward(input, offset, mask,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   deformable_group);
    }
}

std::vector<at::Tensor>
dcn_v3_backward(const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &grad_output,
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int deformable_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v3_cuda_backward(input,
                                    offset,
                                    mask,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v3_cpu_backward(input,
                                    offset,
                                    mask,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    deformable_group);
    }
}

