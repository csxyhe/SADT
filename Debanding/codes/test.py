import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
import cv2
import math
import scipy
import lpips

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_SADT.yml')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)

lpips_matric_alex = lpips.LPIPS(net='alex')

for test_loader in test_loaders:
    # print(len(test_loader))
    test_set_name = test_loader.dataset.opt['name']
    # print(test_set_name)
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    print(dataset_dir)
    util.mkdir(dataset_dir)



    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['lpips'] = []

    for data in test_loader:
        model.feed_data(data)
        img_path = data['GT_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test(s=opt['factor']) 

        visuals = model.get_current_visuals()

        lpips = lpips_matric_alex.forward(torch.clamp(visuals['Denoised'], 0, 1), torch.clamp(visuals['GT'], 0, 1), normalize=True).item()
        test_results['lpips'].append(lpips)

        sr_img = util.tensor2img(visuals['Denoised'])  # uint8
        srgt_img = util.tensor2img(visuals['GT'])  # uint8


        # save images
        suffix = opt['suffix']

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + 'x2.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        
        util.save_img(sr_img, save_img_path)



        # calculate PSNR and SSIM
        gt_img = util.tensor2img(visuals['GT'])

        gt_img = gt_img / 255.
        sr_img = sr_img / 255.


        crop_border = 0
        if crop_border == 0:
            cropped_sr_img = sr_img
            cropped_gt_img = gt_img
            # cropped_draft_img = draft_img
        else:
            cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
        
        psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
        
        # draft_psnr = util.calculate_psnr(cropped_draft_img * 255, cropped_gt_img * 255)
        # draft_ssim = util.calculate_ssim(cropped_draft_img * 255, cropped_gt_img * 255)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        # test_results['draft_psnr'].append(draft_psnr)
        # test_results['draft_ssim'].append(draft_ssim)
        torch.cuda.empty_cache()

        # logger.info(
        #         '{:20s} - draft PSNR: {:.6f} dB; SSIM: {:.6f}, final PSNR: {:.6f} dB; SSIM: {:.6f}.'.
        #     format(img_name, draft_psnr, draft_ssim, psnr, ssim))
        logger.info(
                '{:20s} - PSNR: {:.3f} dB; SSIM: {:.6f}; LPIPS: {:.5f}.'.
            format(img_name, psnr, ssim, lpips))
        # else:
        #     logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}. LR PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim, psnr_lr, ssim_lr))

    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])

    logger.info(
        '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.3f} db; ssim: {:.6f}; lpips: {:.5f}. \n'.format(
        test_set_name, ave_psnr, ave_ssim, ave_lpips))
