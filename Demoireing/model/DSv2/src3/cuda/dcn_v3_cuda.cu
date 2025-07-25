#include <vector>
#include "dcn_v3_im2col_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// THCState *state = at::globalContext().lazyInitCUDA();

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

// [batch gemm]
// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/generic/THCTensorMathBlas.cu


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
                    const int deformable_group)
{
    using scalar_t = float;
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto columns = at::empty({batch, channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    // auto output = at::zeros({batch, channels_out, height_out, width_out}, input.options());

    // // Add biases to output tensor
    // // torch implementation
    // auto ones_T = at::transpose(ones.contiguous(), 3, 1);
    // ones_T = at::mul(ones_T, bias.contiguous());
    // ones_T = at::transpose(ones_T, 3, 1);
    // output = at::add(output, ones_T);

    modulated_deformable_im2col_cuda(c10::cuda::getCurrentCUDAStream(),
                                 input.data_ptr<scalar_t>(),
                                 offset.data_ptr<scalar_t>(),
                                 mask.data_ptr<scalar_t>(),
                                 batch, channels, height, width,
                                 height_out, width_out, kernel_h, kernel_w,
                                 pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                 deformable_group,
                                 columns.data_ptr<scalar_t>());

    // // Scale columns and add to output
    // // torch implementation
    // auto weight_flat = weight.view({channels_out, channels * kernel_h * kernel_w});
    // auto product = at::matmul(weight_flat, columns);
    // output = at::add(output, product.view({batch, channels_out, height_out, width_out}));

    return columns;
}

std::vector<at::Tensor> dcn_v3_cuda_backward(const at::Tensor &input,
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

    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto columns = at::empty({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());

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

        // // Torch implementation
        // auto weight_flat = weight.view({channels_out, channels*kernel_h*kernel_w});
        // weight_flat = at::transpose(weight_flat, 1, 0);

        // auto grad_output_n_flat = grad_output_n.view({channels_out, height_out*width_out});
        // columns = at::matmul(weight_flat, grad_output_n_flat);

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(c10::cuda::getCurrentCUDAStream(),
                                               grad_output_n.data_ptr<scalar_t>(),
                                               input_n.data_ptr<scalar_t>(),
                                               offset_n.data_ptr<scalar_t>(),
                                               mask_n.data_ptr<scalar_t>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data_ptr<scalar_t>(),
                                               grad_mask_n.data_ptr<scalar_t>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(c10::cuda::getCurrentCUDAStream(),
                                         grad_output_n.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data_ptr<scalar_t>());

        // // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        // modulated_deformable_im2col_cuda(c10::cuda::getCurrentCUDAStream(),
        //                                  input_n.data_ptr<scalar_t>(),
        //                                  offset_n.data_ptr<scalar_t>(),
        //                                  mask_n.data_ptr<scalar_t>(),
        //                                  1, channels, height, width,
        //                                  height_out, width_out, kernel_h, kernel_w,
        //                                  pad_h, pad_w, stride_h, stride_w,
        //                                  dilation_h, dilation_w, deformable_group,
        //                                  columns.data_ptr<scalar_t>());


    //    // Torch implementation
    //     auto product = at::matmul(grad_output_n_flat, at::transpose(columns, 1, 0));
    //     grad_weight = at::add(grad_weight, product.view({channels_out, channels, kernel_h, kernel_w}));

    //     // Torch implementation
    //     auto ones_flat = ones.view({height_out*width_out});
    //     product = at::matmul(grad_output_n_flat, ones_flat);
    //     grad_bias = at::add(grad_bias, product);

    }

    return {
        grad_input, grad_offset, grad_mask
    };
}
