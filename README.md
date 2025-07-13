# A Universal Scale-Adaptive Deformable Transformer for Image Restoration across Diverse Artifacts (CVPR 2025)



[Paper Link](https://openaccess.thecvf.com/content/CVPR2025/html/He_A_Universal_Scale-Adaptive_Deformable_Transformer_for_Image_Restoration_across_Diverse_CVPR_2025_paper.html)

> **abstract**: Structured artifacts are semi-regular, repetitive patterns that closely intertwine with genuine image content, making their removal highly challenging. In this paper, we introduce the Scale-Adaptive Deformable Transformer, an network architecture specifically designed to eliminate such artifacts from images. The proposed network features two key components: a scale-enhanced deformable convolution module for modeling scale-varying patterns with abundant orientations and potential distortions, and a scale-adaptive deformable attention mechanism for capturing long-range relationships among repetitive patterns with different sizes and non-uniform spatial distributions. Extensive experiments show that our network consistently outperforms state-of-the-art methods in diverse artifact removal tasks, including image deraining, image demoireing, and image debanding.



The code has been test on 2 NVIDIA RTX 4090 GPUs.

## Requirements

```bash
torch==2.0.0+cu118
timm==0.9.16
einops
```

***The Scale-Adaptive Deformable Attention (SADA) module is partially implemented by CUDA extension now.***

### DSv2

- The sub-module for deformable sampling operation

- Complie Successfully on ubuntu 23.04, cuda_11.8, gcc version 11.4.0, python 3.9.18(conda)

**Install**

```bash
$ CUDA_HOME=your_cuda_path python3 setup.py build develop
```

 ***Another version of the SADA module, which is fully implemented in PyTorch and has a faster speed than CUDA version, will be uploaded soon. Stay tuned!***



## Dataset Download

- Demoireing
  - TIP2018: https://huggingface.co/datasets/zxbsmk/TIP-2018
  - FHDMi: https://drive.google.com/drive/folders/1IJSeBXepXFpNAvL5OyZ2Y1yu4KPvDxN5?usp=sharing or https://pan.baidu.com/s/19LTN7unSBAftSpNVs8x9ZQ (jf2d)
  - LCDMoire: https://competitions.codalab.org/competitions/20166 (the download link can not be available now.)

- Deraining: https://stevewongv.github.io/derain-project.html (the download link can not be available now.)
- Debanding: https://zenodo.org/record/7224906#.ZBUq97dBxPZ



## Pre-trained Models

<table>
<thead>
  <tr>
    <th>Task</th>
    <th>Demoireing</th>
    <th>Demoireing</th>
    <th>Demoireing</th>
    <th>Deraining</th>
    <th>Debanding</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Dataset</td>
    <td>FHDMi  </td>
    <td> TIP18  </td>
    <td> LCDMoire </td>
    <td> SPAD  </td>
    <td>  DID </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td>Download Link</td>
    <td> <a href="https://drive.google.com/file/d/18bebD6Npm5S4h3KStP2vhQ6l3WWZhJN3/view?usp=sharing">Download</a> </td>
    <td> <a href="https://drive.google.com/file/d/1mkMJfZCXnNLWBCgY0o8LVDHv0PdLgsb8/view?usp=sharing">Download</a> </td>
    <td> <a href="https://drive.google.com/file/d/1p_LGJjUqIIPiCumtguGUjLMj7CpwDIhD/view?usp=sharing">Download</a> </td>
    <td> <a href="https://drive.google.com/file/d/1arfwmNYpj8HszfaW0a2HYUmqsFA23Iz0/view?usp=sharing">Download</a> </td>
    <td> <a href="https://drive.google.com/file/d/1ypzrmLbZBomvaSvY8B2ZhXoexVlBdU3R/view?usp=sharing">Download</a> </td>
  </tr>
</tbody>
</table>



## Test

### Demoireing

- Go into the sub-folder:

  ```bash
  $ cd Demoireing
  ```

- Download the corresponding pre-trained model and put it into `out_dir/xxx/exp_light/net_checkpoints` folder.

- `cd model/DSv2`  and compile the CUDA extension into a Python-compatible module
- `cd ../../`
- Download the dataset and open the configuration file `config/xxx_config.yaml`, modify the `TRAIN_DATASET` and `TEST_DATASET`  to your own data path.

- Run the test code:

```bash
$ python test.py --config/xxx_config.yaml
```

### Deraining

- Go into the sub-folder:

  ```bash
  $ cd Deraining
  ```

- Download the corresponding pre-trained model and put it into `experiments\Deraining_SADT_spa\models` folder.

- `cd basicsr/models/archs/DSv2`  and compile the CUDA extension into a Python-compatible module

- `cd ../../../../`, download the dataset and and run the test code:

  ```bash
  $ python test.py --input_dir your_input_path --gt_dir your_gt_path --result_dir your_result_dir
  ```

### Debanding

- Go into the sub-folder:

  ```bash
  $ cd Debanding
  ```

- Download the corresponding pre-trained model and put it into `experiments\SADT_debanding\models` folder.

- Download the dataset. The pristine dataset is not divided, you can divide the image pairs through the same strategy as us: "The dataset comprises 1440 pairs of Full High-Definition (FHD) images. Each image is initially divided into 256 $\times$ 256 patches with a step size of 128. After filtering out pairs which degraded image devoid of banding artifacts, the re maining pairs are divided into training (60%), validation (20%), and test (20%) sets while ensuring all patches from the same FHD image belong to the same set."

- `cd codes` and run the test code:

  ```bash
  $ python test.py -opt options/test/test_SADT.yml
  ```

  

## Training

Similar like Testing. Please see the previous section.



## Citation

If you are interested in this work, please consider citing:

```
@inproceedings{he2025universal,
  title={A Universal Scale-Adaptive Deformable Transformer for Image Restoration across Diverse Artifacts},
  author={He, Xuyi and Quan, Yuhui and Xu, Ruotao and Ji, Hui},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={12731--12741},
  year={2025}
}
```



## Acknowledgement

This code is based on the [DCNv2](https://github.com/CharlesShang/DCNv2), [ESDNet](https://github.com/CVMI-Lab/UHDM) and [DRSformer](https://github.com/cschenxiang/DRSformer). Thanks for their awesome work.
