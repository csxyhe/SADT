#include <vector>
#include "cpu/dcn_v3_im2col_cpu.h"

#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>

//extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu
// modified from the CUDA version for CPU use by Daniel K. Suhendro

at::Tensor
dcn_v3_cpu_forward(const at::Tensor &input,
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
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));
    /*AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");*/

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;


    auto columns = at::empty({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());


    using scalar_t = float;
    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n = mask.select(0, b);
        auto output_n = output.select(0, b);

        modulated_deformable_im2col_cpu(input_n.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group,
                                         columns.data_ptr<scalar_t>());

        //(k * m)  x  (m * n)
        // Y = WC

        output.select(0, b) = at::add(output_n, columns);
    }
    return output;
}

std::vector<at::Tensor> dcn_v3_cpu_backward(const at::Tensor &input,
                                             const at::Tensor &offset,
                                             const at::Tensor &mask,
                                             const at::Tensor &grad_output,
                                             int kernel_h, int kernel_w,
                                             int stride_h, int stride_w,
                                             int pad_h, int pad_w,
                                             int dilation_h, int dilation_w,
                                             int deformable_group)
{

    TORCH_CHECK_ARG(input.is_contiguous(), 1, "input tensor has to be contiguous");
    // TORCH_CHECK_MSG(input.is_contiguous(), 1, "input tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;


    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n = mask.select(0, b);
        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_mask_n = grad_mask.select(0, b);


        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cpu(grad_output_n.data<scalar_t>(),
                                               input_n.data<scalar_t>(),
                                               offset_n.data<scalar_t>(),
                                               mask_n.data<scalar_t>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data<scalar_t>(),
                                               grad_mask_n.data<scalar_t>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cpu(grad_output_n.data<scalar_t>(),
                                         offset_n.data<scalar_t>(),
                                         mask_n.data<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data<scalar_t>());

    }

    return {
        grad_input, grad_offset, grad_mask
    };
}
