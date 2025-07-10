import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser(description='Test on your own images')
parser.add_argument('--input_dir', default='/home/xuyi/data/derain/spa/test/rain/', type=str, help='Directory of input images or path of single image')
parser.add_argument('--gt_dir', default='/home/xuyi/data/derain/spa/test/norain/', type=str, help='Directory of gt images or path of single image')
parser.add_argument('--result_dir', default='./test/output/', type=str, help='Directory for restored results')
parser.add_argument('--pretrained_model', default='./experiments/Deraining_SADT_spa/models/net_g_best.pth', type=str, help='Path to pretrained model')
parser.add_argument('--task', default='Deraining', type=str, help='Deraining', choices=['Deraining'])
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')


args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(task, pret_weights, parameters):
    if task == 'Deraining':
        weights = pret_weights
    return weights, parameters

task    = args.task
ptweights = args.pretrained_model
inp_dir = args.input_dir
gt_dir = args.gt_dir
out_dir = os.path.join(args.result_dir, task)

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
    files = natsorted(files)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

if any([gt_dir.endswith(ext) for ext in extensions]):
    gt_files = [gt_dir]
else:
    gt_files = []
    for ext in extensions:
        gt_files.extend(glob(os.path.join(gt_dir, '*.'+ext)))
    gt_files = natsorted(gt_files)

if len(gt_files) == 0:
    raise Exception(f'No files found at {gt_dir}')

# Get model weights and parameters
parameters = {'in_channels': 3, 'window_size': 8, 'use_bias': True, 'reduction': 4, 'out_channels': 3}
weights, parameters = get_weights_and_parameters(task, ptweights, parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'SADT_arch.py'))
model = load_arch['SADT'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 64

print(f"\n ==> Running {task} with weights {weights}\n ")
avg_psnr = 0.
avg_ssim = 0.

with torch.no_grad():
    for file_, gt_file_ in tqdm(zip(files, gt_files)):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if task == 'Deraining':
            img = load_img(file_)
            gt_img = load_img(gt_file_)

        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if args.tile is None:
            ## Testing on the original resolution image
            restored = model(input_)[0]
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)[0]
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:height,:width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        psnr = calculate_psnr(restored, gt_img, crop_border=0, test_y_channel=True)
        ssim = calculate_ssim(restored, gt_img, crop_border=0, test_y_channel=True)
        avg_psnr += psnr
        avg_ssim += ssim

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        # stx()
        if task == 'Deraining':
            save_img((os.path.join(out_dir, f+'.png')), restored)

    print(f"\nRestored images are saved at {out_dir}")
    avg_psnr = avg_psnr / len(files)
    avg_ssim = avg_ssim / len(files)
    print("AVG_PSNR: {}, AVG_SSIM: {}".format(avg_psnr, avg_ssim))
