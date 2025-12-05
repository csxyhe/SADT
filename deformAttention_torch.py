import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # x.shape = (b, c, h, w)
        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)
        return x


class to_channel_first():
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x.shape = (b, h, w, c)
        return x.permute(0, 3, 1, 2)


class to_channel_last():
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x.shape = (b, c, h, w)
        return x.permute(0, 2, 3, 1)


# ##########################################################################
# ## Refinement Feed-forward Network (FRFN)
class Mlp(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x h x w x c
        bs, h, w, c = x.size()
        # hh = int(math.sqrt(hw))

        # spatial restore
        x = x.permute(0, 3, 1, 2)  # (b, c, h, w)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)

        x = self.linear1(x)  # (b, h, w, 2c)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = x_1.permute(0, 3, 1, 2)  # (b, c, h, w)
        x_1 = self.dwconv(x_1).permute(0, 2, 3, 1)  # (b, h, w, c)
        x = x_1 * x_2

        x = self.linear2(x)
        # x = self.eca(x)

        return x


def flow_warp_parallel(x, flow, interp_mode='bilinear', padding_mode='border'):
    """
    Warp an image or feature map with optical flows.
    根据flow，在x上每个spatial pixel采样K个偏移点

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H_in, W_in).
        flow (torch.Tensor): Optical flow tensor of shape (N, K 2, H, W).
        interp_mode (str): Interpolation mode for grid_sample. Options are 'bilinear' and 'nearest'.
        padding_mode (str): Padding mode for grid_sample. Options are 'zeros', 'border', and 'reflection'.

    Returns:
        torch.Tensor: Warped tensor of shape (N, C, K, H, W).
    """
    N, C, H, W = x.size()
    # Create a grid of (x, y) coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    # 采样中心点(2, H, W) -> (1, 2, H, W) -> (N, 2, H, W)
    grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).expand(N, -1, -1, -1).to(x.device)

    # Normalize the grid to [-1, 1] range
    grid = grid.permute(0, 2, 3, 1).unsqueeze(3)  # Shape: (N, H, W, 1, 2)
    flow = flow.reshape(N, -1, 2, H, W).permute(0, 3, 4, 1, 2)  # (N, H, W, K, 2)
    # (N, H, W K, 2)
    grid = (grid + flow).contiguous().view(N, H, -1, 2)
    grid = 2.0 * grid / torch.tensor([W - 1, H - 1], dtype=torch.float32, device=x.device) - 1.0

    # Warp the input tensor using the grid
    # grid.shape = (N, H, W K, 2)
    # x.shape = (N, C, H, W)
    # warp.shape = (N, C, H, W K)
    warped = F.grid_sample(x, grid, mode=interp_mode, padding_mode=padding_mode, align_corners=True)

    return warped.reshape(N, C, H, W, -1).permute(0, 1, 4, 2, 3)


class MSDSNv3torch(nn.Module):
    def __init__(self, inc=3, kernel_size=3, groups=4):
        """mutli-scale deformable sampling network, different head different scale.

        Args:
            inc (int, optional): _description_. Defaults to 3.
            kernel_size (int, optional): _description_. Defaults to 3.
            padding (int, optional): _description_. Defaults to 1.
            stride (int, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 4.
        """
        super(MSDSNv3torch, self).__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        points_pool = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
        # groups, kk, 2
        base_points = torch.empty(groups, kernel_size ** 2, 2)
        for i in range(groups):
            base_points[i, ...] = torch.tensor(points_pool) * (2*i+1)
        base_points = base_points.view(groups, -1)
        # groups, 1, 1, 1, kk 2
        self.register_buffer('base_offset', base_points[:, None, None, None, :])
        self.conv_offset_list = nn.ModuleList([
            nn.Sequential(PConv(inc),
                          nn.Conv2d(inc, 2 * kernel_size ** 2, 1))
            for _ in range(groups)])
        self.conv_mask = nn.Sequential(PConv(inc),
                                       nn.Conv2d(inc, kernel_size ** 2, 1))
        self.init_offset()

    def init_offset(self):
        for op in self.conv_offset_list:
            for m in op.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for m in self.conv_mask.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        y1, y2 = y.chunk(2, dim=0)
        y1_list = y1.chunk(self.groups, dim=1)
        y2_list = y2.chunk(self.groups, dim=1)

        y1_new_list = list()
        y2_new_list = list()
        base_offset = self.base_offset.to(x.dtype)
        # (groups, B, H, W, kk 2)
        base_offset = base_offset.expand(-1, b, h, w, -1).clone()
        # (groups, B, KK 2, H, W)
        base_offset = base_offset.permute(0, 1, 4, 2, 3)
        # (b, kk, h, w)
        mask = self.conv_mask(x)
        mask = mask[:, None, :, :, :]
        # print("x.dtype is {}".format(x.dtype))
        # print("mask.dtype is {}".format(mask.dtype))
        for i in range(self.groups):
            # (B, KK 2, H, W)
            offset = self.conv_offset_list[i](x) + base_offset[i, ...]
            # (B, C, KK, H, W)
            y1_new_list.append(
                flow_warp_parallel(x=y1_list[i], flow=offset) * mask
            )
            y2_new_list.append(
                flow_warp_parallel(x=y2_list[i], flow=offset) * mask
            )
        y1 = torch.cat(y1_new_list, dim=1)
        y2 = torch.cat(y2_new_list, dim=1)
        y = torch.cat([y1, y2], dim=0)  # (2 * b, c, N, h, w)
        return y


class DeformCenterAttention(nn.Module):
    """
    """

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 stride=1,
                 padding=True,
                 kernel_size=3):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.k_size = kernel_size  # kernel size
        self.stride = stride  # stride
        # self.pat_size = patch_size  # patch size

        self.in_channels = dim  # origin channel is 3, patch channel is in_channel
        self.num_heads = num_heads
        self.head_channel = dim // num_heads
        # self.dim = dim # patch embedding dim
        # it seems that padding must be true to make unfolded dim matchs query dim h*w*ks*ks
        self.pad_size = kernel_size // 2 if padding is True else 0  # padding size
        self.pad = nn.ZeroPad2d(self.pad_size)  # padding around the input
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        # self.unfold = nn.Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)
        # 改用q来决定kv要采样哪些位置
        self.dsn = MSDSNv3torch(inc=dim, kernel_size=kernel_size, padding=self.pad_size, stride=1, groups=num_heads)

        self.qkv_bias = qkv_bias
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_dwconv = DWConv(dim=dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        # x = x.reshape(B, H, W, C)
        assert C == self.in_channels

        self.num_patch = H * W  # 切出来的patch的数量

        qkv = self.qkv_proj(x)
        # (B, H, W, 3 * C)
        qkv = self.qkv_dwconv(qkv)
        q = qkv[:, :, :, :C]
        kv = qkv[:, :, :, C:]

        # # (2, B, NumHeads, HeadsC, H, W)
        kv = kv.reshape(B, H, W, 2, self.num_heads, self.head_channel).permute(3, 0, 4, 5, 1, 2)

        # kv = self.pad(kv)  # (2, B, NumH, HeadC, H, W)
        # H, W = H + self.pad_size * 2, W + self.pad_size * 2

        # unfold plays role of conv2d to get patch data
        kv = kv.reshape(2 * B, -1, H, W)  # (2 * B, C, pad_H, pad_W)
        kv = self.dsn(q.permute(0, 3, 1, 2), kv)  # (2 * B, C, N, h, w)
        # kv = self.unfold(kv)

        # # (B, NumHeads, H, W, HeadC)
        q = q.reshape(B, H, W, self.num_heads, self.head_channel).permute(0, 3, 1, 2, 4)
        # # q = self.pad(q).permute(0, 1, 3, 4, 2)  # (B, NumH, H, W, HeadC)
        # # query need to be copied by (self.k_size*self.k_size) times
        q = q.unsqueeze(dim=4)
        q = q * self.scale
        # # if stride is not 1, q should be masked to match ks*ks*patch
        # # ...

        kv = kv.reshape(2, B, self.num_heads, self.head_channel, self.k_size ** 2,
                        self.num_patch)  # (2, B, NumH, HC, ks*ks, NumPatch)
        kv = kv.permute(0, 1, 2, 5, 4, 3)  # (2, B, NumH, NumPatch, ks*ks, HC)
        k, v = kv[0], kv[1]

        # (B, NumH, NumPatch, 1, HeadC)
        q = q.reshape(B, self.num_heads, self.num_patch, 1, self.head_channel)
        attn = (q @ k.transpose(-2, -1))  # (B, NumH, NumPatch, ks*ks, ks*ks)
        attn = self.softmax(attn)  # softmax last dim
        attn = self.attn_drop(attn)

        out = (attn @ v).squeeze(3)  # (B, NumH, NumPatch, HeadC)
        out = out.permute(0, 2, 1, 3).reshape(B, H, W, C)  # (B, Ph, Pw, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        # out = out.reshape(B, -1, C)
        return out


class SADAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DeformCenterAttention(
            dim=dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

