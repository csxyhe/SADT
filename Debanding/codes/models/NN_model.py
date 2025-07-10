import logging
from collections import OrderedDict
from re import L

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from .SADT_archs import SADT



logger = logging.getLogger('base')


class NN_Model(BaseModel):
    def __init__(self, opt):
        super(NN_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        logger_opt = opt['logger']
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.logger_opt = logger_opt


        opt_net = opt['network_G']

        self.netG = SADT(in_channels=opt_net['in_nc'], out_channels=opt_net['out_nc']).to(self.device)


       
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'])
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        # self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT
        # print(self.real_H.shape)
        self.noisy_H = data['Noisy'].to(self.device)  # Noisy

    # Real Input
    def feed_test_data(self, data):
        self.noisy_H = data.to(self.device)  # Noisy

    
    # 非可逆网络用
    def optimize_parameters(self, step):
        criterion = nn.L1Loss()
        self.optimizer_G.zero_grad()
        out_big, out_mid, out_small = self.netG(x=self.noisy_H)
        self.output = out_big
        
        mid_realH = F.interpolate(self.real_H, scale_factor=0.5, mode='bilinear')
        small_realH = F.interpolate(self.real_H, scale_factor=0.25, mode='bilinear')
        # L1 loss
        content_loss = criterion(self.output, self.real_H) + criterion(out_mid, mid_realH) + criterion(out_small, small_realH)
        
        realH_fft = torch.fft.fft2(self.real_H)
        mid_realH_fft = torch.fft.fft2(mid_realH)
        small_realH_fft = torch.fft.fft2(small_realH)

        out_big_fft = torch.fft.fft2(out_big)
        out_mid_fft = torch.fft.fft2(out_mid)
        out_small_fft = torch.fft.fft2(out_small)

        freq_loss = criterion(out_big_fft.real, realH_fft.real) + criterion(out_big_fft.imag, realH_fft.imag) + \
                    criterion(out_mid_fft.real, mid_realH_fft.real) + criterion(out_mid_fft.imag, mid_realH_fft.imag) + \
                    criterion(out_small_fft.real, small_realH_fft.real) + criterion(out_small_fft.imag, small_realH_fft.imag)


        loss = content_loss + 0.1 * freq_loss
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()
        self.log_dict['cont_loss'] = content_loss.item()
        self.log_dict['freq_loss'] = freq_loss.item()
        self.log_dict['loss'] = loss.item()

        if step % self.logger_opt['save_checkpoint_freq'] == 0 or step == (self.logger_opt['save_checkpoint_freq'] // 2):
            self.save(step)


    def test(self, s=64):
        self.input = self.noisy_H
        h, w = self.input.size(2), self.input.size(3)
        if h % s == 0:
            pad_h = 0
        else:
            pad_h = s - h % s
        if w % s == 0:
            pad_w = 0
        else:
            pad_w = s - w % s
        self.input = F.pad(self.input, pad=(0, pad_w, 0, pad_h), mode='reflect')            
        self.netG.eval()
        with torch.no_grad():
            out_big, _, _ = self.netG(self.input)
            if pad_h is not None:
                h, w = out_big.shape[2], out_big.shape[3]
                self.fake_H = out_big[:, :, :h - pad_h, :w - pad_w]
        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Denoised'] = self.fake_H.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['Noisy'] = self.noisy_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

