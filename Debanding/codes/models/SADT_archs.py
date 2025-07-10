import torch.nn as nn
import torch
import einops
import torch.nn.functional as F
from .deformAttention import SADAttentionBlock, PConv
import torchvision



class myLayerNorm(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=channel_in)

    def forward(self, x):
        # x.shape = (b, c, h, w)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, channels, height, width = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n c (gh fh) (gw fw) -> n c (gh gw) (fh fw)",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n c (gh gw) (fh fw) -> n c (gh fh) (gw fw)",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x




def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class DeformConv(nn.Module):
    def __init__(self, inc, outc, ksize):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=ksize, stride=1, padding=ksize // 2) #原卷积

        self.conv_offset = nn.Sequential(PConv(inc),
                                        nn.Conv2d(inc, 2 * (ksize ** 2), 1, bias=True))
        self.conv_mask = nn.Sequential(PConv(inc),
                                        nn.Conv2d(inc, (ksize ** 2), 1, bias=False))
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.conv_offset.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.conv_mask.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)
    
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) #保证在0到1之间
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                             mask=mask, padding=(1, 1))
        return out

class DeformSPPFBlockGatingUnit(nn.Module):
    def __init__(self, dim, block_size):
        super(DeformSPPFBlockGatingUnit, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // 4, 3, 1, 1)
        self.spatialProj0 = BlockGatingUnit(dim // 4, block_size)
        # self.c1 = nn.MaxPool2d(5, 1, 2)
        self.c1 = nn.Sequential(DeformConv(dim // 4, dim // 4, 3),
                                nn.ReLU(inplace=True))
        self.spatialProj1 = BlockGatingUnit(dim // 4, block_size)
        # self.c2 = nn.MaxPool2d(5, 1, 2)
        self.c2 = nn.Sequential(DeformConv(dim // 4, dim // 4, 3),
                                nn.ReLU(inplace=True))
        self.spatialProj2 = BlockGatingUnit(dim // 4, block_size)
        # self.c3 = nn.MaxPool2d(5, 1, 2)
        self.c3 = nn.Sequential(DeformConv(dim // 4, dim // 4, 3),
                                nn.ReLU(inplace=True))
        self.spatialProj3 = BlockGatingUnit(dim // 4, block_size)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        out1 = self.spatialProj0(x)

        x = self.c1(x)
        out2 = self.spatialProj1(x)

        x = self.c2(x)
        out3 = self.spatialProj2(x)

        x = self.c3(x)
        out4 = self.spatialProj3(x)

        out = torch.cat((out1, out2, out3, out4), 1)

        out = self.conv2(out)

        return out



class BlockGatingUnit(nn.Module):
    """gMLP模块中门电路 + spatial proj的内容 (conv版本)，加速版本！"""
    def __init__(self, dim=6, block_size=8, use_bias=True):
        super(BlockGatingUnit, self).__init__()
        n = block_size**2
        k = block_size
        self.k = k
        self.conv = nn.Conv2d(1, k * k, k, bias=True, stride=k)
        self.up = nn.PixelShuffle(k)
        self.dim = dim

    def forward(self, x):
        # x.shape = b, c, gh*gw, fh*fw
        shortcut = x
        x = F.conv2d(x, weight=self.conv.weight.repeat(self.dim, 1, 1, 1), 
                    bias=self.conv.bias.repeat(self.dim), stride=self.k, groups=self.dim)
        x = self.up(x)
        return shortcut * x



# ##########################################################################
# ## Refinement Feed-forward Network (FRFN)
class ChannelMLP(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super(ChannelMLP, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim*2),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Conv2d(hidden_dim, dim, 1)
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv 
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # (b, c, h, w)
        bs, c, h, w = x.size()


        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = x.permute(0, 2, 3, 1) # (b, h, w, c)

        x = self.linear1(x) # (b, h, w, 2c)
        #gate mechanism
        x_1,x_2 = x.chunk(2,dim=-1) # (b, h, w, c)

        x_1 = x_1.permute(0, 3, 1, 2) # (b, c, h, w)
        x_1 = self.dwconv(x_1)
        x = x_1 * x_2.permute(0, 3, 1, 2)
        
        x = self.linear2(x)
        # x = self.eca(x)

        return x


class DeformSPPFSpatialMLP(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        self.channelProj1 = conv(dim, dim, 1)
        self.acti = nn.GELU()
        self.sppfSpatialProj = DeformSPPFBlockGatingUnit(dim, block_size)
        self.channelProj2 = conv(dim, dim, 1)
    
    def forward(self, x):
        x = self.acti(self.channelProj1(x))
        x = self.channelProj2(self.sppfSpatialProj(x))
        return x


class ScaleEnhancedDC(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.ln1 = myLayerNorm(dim)
        self.spatialMLP = DeformSPPFSpatialMLP(dim, window_size)
        self.ln2 = myLayerNorm(dim)
        self.channelMLP = ChannelMLP(dim)

    def forward(self, x):
        shortcut = x
        x = self.spatialMLP(self.ln1(x)) 
        x = x + shortcut
        shortcut = x
        x = self.channelMLP(self.ln2(x)) + shortcut
        return x


class DownSample_2(nn.Module):
    """使用步长为2，大小为4*4的卷积核来做下采样"""

    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.convDown = nn.Conv2d(in_channels=channel_in, out_channels=channel_out,kernel_size=(4,4),stride=2,padding=1)

    def forward(self, x):
        return self.convDown(x)


class UpSample(nn.Module):
    """使用大小为2*2，步长超参数设置为2的反卷积来做上采样"""

    def __init__(self, features):
        super().__init__()
        self.convUp = nn.ConvTranspose2d(in_channels=features, out_channels=features, kernel_size=(2, 2), stride=2)

    def forward(self, x):
        return self.convUp(x)


class ChangeChannelsConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=use_bias)

    def forward(self, x):
        return self.conv(x)


# 给coarsest scale的encoder / decoder / bottleneck用
class DSAXcoder(nn.Module):
    def __init__(self, features, window_size=8, depth=4, num_heads=4, csff=False):
        """_summary_

        Args:
            features (_type_): _description_
            window_size (int, optional): _description_. Defaults to 8.
            depth (int, optional): _description_. Defaults to 4.
            num_heads (int, optional): _description_. Defaults to 4.
            csff (bool, optional): 标志是否有cross-stage feature fusion,只有encoder才有可能设置为True. Defaults to False.
        """
        super().__init__()

        self.csff = csff
        if csff:
            self.transfer_prev_enc = nn.Conv2d(features, features, kernel_size=1, bias=False)
            self.transfer_prev_dec = nn.Conv2d(features, features, kernel_size=1, bias=False)
 
       
        self.firstSPPF = ScaleEnhancedDC(features, window_size)
        self.blocks = nn.ModuleList([
            SADAttentionBlock(dim=features, num_heads=num_heads)
            for i in range(depth - 2)])
        self.lastSPPF = ScaleEnhancedDC(features, window_size)
        self.depth = depth

    def forward(self, x, prev_enc=None, prev_dec=None):
        """
        Args:
            x (_type_): _description_
        """
        if self.csff:
            x = x + self.transfer_prev_enc(prev_enc) + self.transfer_prev_dec(prev_dec)
        shortcut = x
        x = self.firstSPPF(x)
        if self.depth > 2:
            x = x.permute(0, 2, 3, 1)
            for blk in self.blocks:
                x = blk(x)
            x = x.permute(0, 3, 1, 2) # (b, c, h, w)
       
        x = self.lastSPPF(x)
        return x + shortcut



class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel):
        """x1: output of the above Encoder, 
            x2: output of SCM
        """
        super(FAM, self).__init__()
        # self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
        self.merge = nn.Sequential(PConv(channel), nn.Conv2d(channel, channel, 1))

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class ReSample(nn.Module):  #input shape: n,c,h,w.    c-->2c
    """features统一维度用的, 无论扩大缩小都用这个, 扩大: ratio > 1, 缩小: ratio < 1
        即能统一spatial resolution, 又不使用过多的参数量
        Upsample features given a ratio > 0.   
    """
    def __init__(self, in_features, out_features, b=0, ratio=2., use_bias=True):
        super().__init__()
        self.ratio = ratio
        self.Conv_0 = nn.Conv2d(in_features, out_features,kernel_size=(1,1),stride=1,bias=use_bias)
    def forward(self, x):
        n,c,h,w = x.shape
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = self.Conv_0(x)
        return x

##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
##### 传入一个list
class SKFF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        # v1, v2
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        # v1, v2 -> s1, s2， 相当于互补
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V  

class SADT(nn.Module):
    def __init__(self, in_channels=3, window_size=8,
                 use_bias=True, factor=2, reduction=4, out_channels=3):
        super().__init__()
        # 各个尺度level规定的通道数
        features1 = 32
        features2 = 64
        features3 = 128

        ### encoder之外
        # SCM模块：get shallow features
        self.SCM_list = nn.ModuleList([
            SCM(out_plane=features1),
            SCM(out_plane=features2),
            SCM(out_plane=features3)
        ])

        # FAM模块：融合当前encoder的输入特征和上一个encoder的输出特征
        self.fam_list = nn.ModuleList([
            FAM(features2),
            FAM(features3)
        ])

        ### encoder
        self.encoderBlock_list = nn.ModuleList([
            DSAXcoder(features=features1, depth=2),
            DSAXcoder(features=features2, depth=6),
            DSAXcoder(features=features3, depth=6)
        ])
        # downsample模块
        self.ds_list = nn.ModuleList([
            DownSample_2(features1, features2),
            DownSample_2(features2, features3),
            DownSample_2(features3, features3)
        ])


        # bottleneck
        self.bottleneckBlock = DSAXcoder(features=features3, depth=6)

        ### decoder
        self.convAdjust_list = nn.ModuleList([
            ChangeChannelsConv(in_channels=features3 + features3, out_channels=features3, use_bias=use_bias),
            ChangeChannelsConv(in_channels=features2 + features3, out_channels=features2),
            ChangeChannelsConv(in_channels=features1 + features2, out_channels=features1)
        ])

        self.decoderBlock_list = nn.ModuleList([
            DSAXcoder(features=features3, depth=6),
            DSAXcoder(features=features2, depth=6),
            DSAXcoder(features=features1, depth=2)
        ])


        self.convToImage_0 = nn.Conv2d(in_channels=features1, out_channels=out_channels, kernel_size=3,
                    padding=1, bias=use_bias)

        self.convToImage_1 = nn.Conv2d(in_channels=features2, out_channels=out_channels, kernel_size=3,
                    padding=1, bias=use_bias)

        self.convToImage_2 = nn.Conv2d(in_channels=features3, out_channels=out_channels, kernel_size=3,
                    padding=1, bias=use_bias)

        # upsample模块
        self.us_list = nn.ModuleList([
            UpSample(features3),
            UpSample(features3),
            UpSample(features2)
        ])


        # skip connection
        self.adjustbf_skip_0 = nn.ModuleList([
            ReSample(features1, features1, ratio=1), # 0->0
            ReSample(features2, features1, ratio=2), # 1->0
        ])

        self.adjustbf_skip_1 = nn.ModuleList([
            ReSample(features2, features2, ratio=1),   # 1->1
            ReSample(features3, features2, ratio=2)    # 2->1
        ])


        self.skip_list = nn.ModuleList([
            SKFF(features1, height=2, reduction=4),
            SKFF(features2, height=2, reduction=4)
        ])


        self.adjustbf_final_0 = nn.ModuleList([
            ReSample(features2, features1, ratio=2),  # 1->0
            ReSample(features1, features1, ratio=1)   # 0->0
        ])

        self.adjustbf_final_1 = nn.ModuleList([
            ReSample(features3, features2, ratio=2),    # 2->1
            ReSample(features2, features2, ratio=1),   # 1->1
        ])

        self.final_fuse_0 = SKFF(features1, height=2, reduction=8)
        self.final_fuse_1 = SKFF(features2, height=2, reduction=8)

    def forward(self, x):
        # x is the degraded big-scale image
        # make multi input
        big = x
        mid = F.interpolate(big, scale_factor=0.5, mode='bilinear')
        small = F.interpolate(mid, scale_factor=0.5, mode='bilinear')

        # 储存各Encoder的原始输出, 备用
        encs_for_skip = list()
        # encoder1
        x = self.SCM_list[0](big)
        x = self.encoderBlock_list[0](x)
        encs_for_skip.append(x)
        x = self.ds_list[0](x)

        # encoder2
        midf = self.SCM_list[1](mid)
        x = self.fam_list[0](x, midf)
        x = self.encoderBlock_list[1](x)
        encs_for_skip.append(x)
        x = self.ds_list[1](x)

        # encoder3
        smallf = self.SCM_list[2](small)
        x = self.fam_list[1](x, smallf)
        x = self.encoderBlock_list[2](x)
        encs_for_skip.append(x)
        x = self.ds_list[2](x)


        # skip connection
        skipSmall = encs_for_skip[2]


        skip11 = self.adjustbf_skip_1[0](encs_for_skip[1])
        skip21 = self.adjustbf_skip_1[1](encs_for_skip[2])
        skipMid = self.skip_list[1]([skip11, skip21])

        skip00 = self.adjustbf_skip_0[0](encs_for_skip[0])
        skip10 = self.adjustbf_skip_0[1](encs_for_skip[1])
        skipBig = self.skip_list[0]([skip00, skip10])


        encs_for_skip.clear()

        x = self.bottleneckBlock(x)

        x = self.us_list[0](x)

        features_for_final = list()

        # decoder3
        x = torch.cat([x, skipSmall], dim=1)
        x = self.convAdjust_list[0](x)
        x = self.decoderBlock_list[0](x)
        features_for_final.append(x)


        x = self.us_list[1](x)

        # decoder2
        x = torch.cat([x, skipMid], dim=1)
        x = self.convAdjust_list[1](x)
        x = self.decoderBlock_list[1](x)
        features_for_final.append(x)

        x = self.us_list[2](x)

        # decoder1
        x = torch.cat([x, skipBig], dim=1)
        x = self.convAdjust_list[2](x)
        x = self.decoderBlock_list[2](x)
        features_for_final.append(x)

        r2 = self.convToImage_2(features_for_final[0])
        outsmall = small + r2

        r1 = self.final_fuse_1([self.adjustbf_final_1[0](features_for_final[0]), self.adjustbf_final_1[1](features_for_final[1])])
        outmid = mid + self.convToImage_1(r1)
        
        r0 = self.final_fuse_0([self.adjustbf_final_0[0](r1), self.adjustbf_final_0[1](features_for_final[2])])
        outbig = big + self.convToImage_0(r0)
        
        features_for_final.clear()


        return outbig, outmid, outsmall