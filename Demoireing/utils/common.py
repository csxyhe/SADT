import cv2
from datetime import datetime
import logging
import math
import numpy as np
import os
import random
from shutil import get_terminal_size
import sys
import time
from thop import profile, clever_format
import torch
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from math import exp
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AdaptiveLRScheduler(_LRScheduler):
    def __init__(self, optimizer, threshold, factor=0.5, min_lr=1e-6, last_epoch=-1):
        """logic: Total 150 epoch. The learning rate is initialized to be 2e-4. The validation was conducted after 
        every 5 training epochs. If the decrease in the validation loss was lower than `threshold`, the learning rate was 
        halved. When the learning rate became lower than 1e-6, the training procedure was completed.
        
        threshold值需要看全过程的余弦退火算法的运行效果来进行设置

        Args:
            threshold (float): if the decrease of current validation loss (for the best loss) is lower than `threshold`, do self._reduce_lr() 
            factor (_type_): when decreasing lr, do new_lr = lr * factor.
            min_lr (_type_): if lr is lower than min_lr, the training procedure is completed.
            last_epoch (int, optional): _description_. Defaults to -1.
        """
        self.threshold = threshold
        self.factor = factor
        self.min_lr = min_lr
        self.num_decrease_times = 0 # store the times of reducing lr
        self.best_loss = float('inf') # Initialize with a very large value
        self.last_epoch = last_epoch
        super(AdaptiveLRScheduler, self).__init__(optimizer, last_epoch)

    def step(self, current_loss=None):
        if current_loss is not None:
            # 仍有较大的拟合空间
            if self.best_loss - current_loss > self.threshold:
                self.best_loss = current_loss
            # loss仍在下降，但拟合空间不足
            elif self.best_loss - current_loss > 0:
                self.best_loss = current_loss
                self.num_decrease_times += 1
                self._reduce_lr()
            # 已经完全过拟合或者其他原因
            else:
                self.num_decrease_times += 1
                self._reduce_lr()

        super(AdaptiveLRScheduler, self).step()

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            if new_lr < param_group['lr']:
                param_group['lr'] = new_lr
                print(f'Reducing learning rate to {new_lr:.2e}')

    def get_lr(self):
        return [max(base_lr * (self.factor ** self.num_decrease_times), self.min_lr) for base_lr in self.base_lrs]

    def is_finished(self):
        return all(lr <= self.min_lr for lr in self.get_lr())


# optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
# lr_scheduler = AdaptiveLRScheduler(optimizer, threshold=0.01, factor=0.5, min_lr=1e-6, last_epoch=-1)

# for epoch in range(150):
#     train(...)  # 训练代码
#     if epoch % 5 == 0:
#         val_loss = validate(...)  # 验证代码
#         lr_scheduler.step(current_loss=val_loss)
    
#     if lr_scheduler.is_finished():
#         print("Training completed due to low learning rate.")
#         break


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        self.T_cur = 0 if last_epoch < 0 else last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                                  "please use `get_last_lr()`.", UserWarning)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    def step(self, epoch=None):
        """Step could be called after every batch update
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o
            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self
            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self
        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    
    
    return img_np.astype(out_type)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad

def calculate_cost(model, input_size=(1,3,224,224)):
    input_ = torch.randn(input_size).cuda()
    macs, params = profile(model, inputs=(input_, ))
    macs, params = clever_format([macs, params], "%.3f")
    logging.warning("MACs:" + macs + ", Params:" + params)

def img_pad(x, w_pad, h_pad, w_odd_pad, h_odd_pad):
    '''
    Here the padding values are determined by the average r,g,b values across the training set
    in FHDMi dataset. For the evaluation on the UHDM, you can also try the commented lines where
    the mean values are calculated from UHDM training set, yielding similar performance.
    '''
    x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3827)
    x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.4141)
    x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3912)
    # x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.5165)
    # x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4952)
    # x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4695)
    y = torch.cat([x1, x2, x3], dim=1)

    return y

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

# Classes to re-use window
class SSIM(torch.nn.Module):
    """
    Fast pytorch implementation for SSIM, referred from
    "https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py"
    """
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        
class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        psnr = -10*torch.log10(torch.mean((img1-img2)**2))
        
        return psnr
        


