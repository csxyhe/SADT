import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from model.model import model_fn_decorator
from model.SADT_archs import SADT
from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
import lpips
from config.config import args
from utils.metric import create_metrics


def val_epoch(args, ValImgLoader, model, model_fn_val, net_metric, epoch, save_path):
    save_path = save_path + '/' + '%04d' % epoch
    mkdir(save_path)
    tbar = tqdm(ValImgLoader)
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0

    for batch_idx, data in enumerate(tbar):
        cur_psnr, cur_ssim, cur_lpips, cur_time = model_fn_val(args, data, model, save_path, net_metric)

        total_psnr += cur_psnr
        avg_val_psnr = total_psnr / (batch_idx + 1)
        total_ssim += cur_ssim
        avg_val_ssim = total_ssim / (batch_idx + 1)
        total_lpips += cur_lpips
        avg_val_lpips = total_lpips / (batch_idx + 1)
        desc = 'Validation: Epoch %d, Avg. LPIPS = %.4f, Avg. PSNR = %.4f and SSIM = %.4f' % (
            epoch, avg_val_lpips, avg_val_psnr, avg_val_ssim)
        tbar.set_description(desc)
        tbar.update()

    return avg_val_psnr, avg_val_ssim, avg_val_lpips

def train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch, iters, lr_scheduler):
    """
    Training Loop for each epoch
    """
    tbar = tqdm(TrainImgLoader)
    total_loss = 0
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):
        loss = model_fn(args, data, model, iters)
        # backward and update
        optimizer.zero_grad()
        loss.backward()
        if epoch > args.T_0:
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        iters += 1
        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx+1)
        desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, lr, avg_train_loss)
        tbar.set_description(desc)
        tbar.update()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    # the learning rate is adjusted after each epoch
    lr_scheduler.step()

    return lr, avg_train_loss, iters



def init():
    # Make dirs
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    args.VAL_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'VAL_DIR')
    mkdir(args.LOGS_DIR)
    mkdir(args.NETS_DIR)
    mkdir(args.VISUALS_DIR)
    mkdir(args.VAL_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.GPU_ID))
    device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # summary writer
    logger = SummaryWriter(args.LOGS_DIR)
    
    return logger, device

def load_checkpoint(model, optimizer, load_epoch):
    load_dir = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    model_state_dict = torch.load(load_dir)['state_dict']
    model.load_state_dict(model_state_dict)
    optimizer_dict = torch.load(load_dir)['optimizer']
    optimizer.load_state_dict(optimizer_dict)
    learning_rate = torch.load(load_dir)['learning_rate']
    iters = torch.load(load_dir)['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))

    return learning_rate, iters

def load_checkpoint_latest(model, optimizer):
    load_dir = args.NETS_DIR + '/checkpoint' + '_latest' + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    model_state_dict = torch.load(load_dir)['state_dict']
    model.load_state_dict(model_state_dict)
    optimizer_dict = torch.load(load_dir)['optimizer']
    optimizer.load_state_dict(optimizer_dict)
    learning_rate = torch.load(load_dir)['learning_rate']
    iters = torch.load(load_dir)['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))

    return learning_rate, iters

def main():
    logger, device = init()
    # create model
    model = SADT().to(device)
    if torch.cuda.device_count() > 1:
        print(f"There are {torch.cuda.device_count()} available GPUs!")
        model = nn.DataParallel(model, device_ids=args.GPU_ID)
    

    compute_metrics = create_metrics(args, device=device)

    # create optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.BASE_LR}], lr=args.BASE_LR, betas=(0.9, 0.999))
    learning_rate = args.BASE_LR
    iters = 0
    # resume training
    if args.LOAD_EPOCH:
        learning_rate, iters = load_checkpoint(model, optimizer, args.LOAD_EPOCH)
    # create loss function
    loss_fn = multi_VGGPerceptualLoss(lam=args.LAM, lam_p=args.LAM_P).to(device)

    # create learning rate scheduler
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
                                               last_epoch=args.LOAD_EPOCH - 1)

    # create training function
    model_fn = model_fn_decorator(loss_fn=loss_fn, device=device)
    model_fn_val = model_fn_decorator(loss_fn=loss_fn, device=device, mode='test')

    # create dataset
    train_path = args.TRAIN_DATASET
    val_path = args.TEST_DATASET

    TrainImgLoader = create_dataset(args, data_path=train_path, mode='train')
    ValImgLoader = create_dataset(args, data_path=val_path, mode='test')

    # start training
    print("****start traininig!!!****")
    avg_train_loss = 0
    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        learning_rate, avg_train_loss, iters = train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch,
                                                           iters, lr_scheduler)
        logger.add_scalar('Train/avg_loss', avg_train_loss, epoch)
        logger.add_scalar('Train/learning_rate', learning_rate, epoch)

        # Save the network per ten epoch
        if epoch % args.VAL_TIME == 0:
            savefilename = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % epoch + '.tar'
            torch.save({
                'learning_rate': learning_rate,
                'iters': iters,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
            }, savefilename)
            avg_val_psnr, avg_val_ssim, avg_val_lpips = val_epoch(args, ValImgLoader, model, model_fn_val,
                                                                                compute_metrics, epoch, args.VAL_DIR)

            logger.add_scalar('Validation/avg_psnr', avg_val_psnr, epoch)
            logger.add_scalar('Validation/avg_ssim', avg_val_ssim, epoch)
            logger.add_scalar('Validation/avg_lpips', avg_val_lpips, epoch)
        # Save the latest model
        savefilename = args.NETS_DIR + '/checkpoint' + '_' + 'latest.tar'
        torch.save({
            'learning_rate': learning_rate,
            'iters': iters,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()
        }, savefilename)


if __name__ == '__main__':
    main()
