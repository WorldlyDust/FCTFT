
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x,is_cuda=False):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
 #########################需要加上cuda########################
    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float()
    if is_cuda==True:
        h=h.cuda()
###############################################
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x,is_cuda=True)



class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        # self.to_q = nn.Conv2d(dim, dim_head * heads, 3, 1, 1, bias=False, groups=dim)
        # self.to_k  = nn.Conv2d(dim, dim_head * heads, 3, 1, 1, bias=False, groups=dim)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        # q_inp = self.to_q(x_in.permute(0, 3, 1, 2))  # [b,h,w,c]-->[b,c,h,w]
        # q_inp = q_inp.permute(0, 2, 3, 1).reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # out_p2 = self.pos_emb(x_in.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # out = out_c + out_p+out_p2
        out=out_c+out_p
        return out

###############wave##################################
# class FeedForward(nn.Module):
#     def __init__(self, dim, mult=4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
#             GELU(),
#             nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
#             GELU(),
#             nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
#         )
#         self.DWT=DWT()
#         self.IDWT=IWT()
#         self.dwt_res=nn.Sequential(nn.Conv2d(dim,dim,1,1),GELU(),nn.Conv2d(dim,dim,1,1))
#     def forward(self, x):
#         """
#         x: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#         x=x.permute(0,3,1,2)
#         # out = self.net(x.permute(0, 3, 1, 2))
#         # out1=self.net(x)
#         _, _, H, W = x.shape
#
#         y_dwt=self.DWT(x)
#         # y = torch.fft.rfft2(x, norm="backward")
#         y_t=self.dwt_res(y_dwt)
#         y1=self.IDWT(y_t)
#         # out2=y1+out1
#         out=y1+x
#         return out.permute(0, 2, 3, 1)
#         # return out2.permute(0, 2, 3, 1)
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, mult=4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
#             GELU(),
#             nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
#             GELU(),
#             nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
#         )
#         self.fft_res=nn.Sequential(nn.Conv2d(2*dim,2*dim,1,1),nn.GELU(),nn.Conv2d(2*dim,2*dim,1,1))
#     def forward(self, x):
#         """
#         x: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#         x=x.permute(0,3,1,2)
#         # out = self.net(x.permute(0, 3, 1, 2))
#         out1=self.net(x)
#         _, _, H, W = x.shape
#         dim = 1
#         y = torch.fft.rfft2(x, norm="backward")
#         y_imag = y.imag
#         y_real = y.real
#         y_f = torch.cat([y_real, y_imag], dim=dim)
#         y = self.fft_res(y_f)
#         y_real, y_imag = torch.chunk(y, 2, dim=dim)
#         # 把实部，虚部组合起来。
#         y = torch.complex(y_real, y_imag)
#         y = torch.fft.irfft2(y, s=(H, W), norm="backward")
#         out2=y+out1
#         return out2.permute(0, 2, 3, 1)

#########################################
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)



#######################################333
class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class TimesDowning(nn.Module):
    def __init__(self, C,Times):
        super(TimesDowning, self).__init__()
        if Times==2:
        # 使用卷积进行2倍的下采样，通道数不变
            self.Down = nn.Sequential(nn.Conv2d(C, C, 3, 2, 1),nn.LeakyReLU())

        if Times==4:
            self.Down = nn.Sequential(nn.Conv2d(C, C, 5, 4, 2), nn.LeakyReLU())
    def forward(self, x):
        return self.Down(x)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)
#################

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             # nn.Sigmoid()
#         )
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y=self.avg_pool(x)
#         y=y.view(b,c)
#         y=self.fc(y)
#         y=y.view(b,c,1,1)
#         out=x*y.expand_as(x)
#         return out

###########
class MST11(nn.Module):
    def __init__(self,in_dim=31,out_dim=31, dim=31, stage=2, num_blocks=[1,1,1]):
        super(MST11, self).__init__()
        self.dim = dim
        self.stage = stage
        # self.SE=SELayer(31)
        filters=[31,62,124,248]
        self.CatChannels=filters[0]
        self.MSAB=MSAB(2*in_dim,in_dim,2,1)
        self.MSAB1=MSAB(in_dim,in_dim,1,1)
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),#downsize
            ]))
            dim_stage *= 2
        self.h1_PT_hd2 = TimesDowning(in_dim, 2)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h3_UT_hd2=nn.ConvTranspose2d(filters[2],filters[1],stride=2, kernel_size=2, padding=0, output_padding=0)
        self.h3_UT_hd2_conv=nn.Conv2d(5*in_dim,2*in_dim,1,1,0)
        # self.h3_UT_hd2_conv_test= nn.Conv2d(3* in_dim, 2 * in_dim, 1, 1, 0)
        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])
        dim_stage = dim_stage // 2
        self.decoder_layers = nn.ModuleList([])
        self.hd3_UT_hd1= nn.ConvTranspose2d(filters[2], filters[0], stride=4, kernel_size=4, padding=0, output_padding=0)
        self.hd3_UT_hd1_conv= nn.Conv2d(3*in_dim, in_dim, 1, 1, bias=False)

        # self.hd3_UT_hd1_conv_test= nn.Conv2d(2* in_dim, in_dim, 1, 1, bias=False)

        self.hd2_UT_hd1 = nn.ConvTranspose2d(filters[1], filters[0], stride=2, kernel_size=2, padding=0,output_padding=0)
        for i in range(stage-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2
        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)
        ###########################

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = self.embedding(x) #[1,31,256,256]
        # Encoder
        fea_encoder = []
        for (MSAB,  FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
        # Bottleneck
        fea = self.bottleneck(fea)
        out=fea
        # Decoder
        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(fea_encoder[0]))
        hd3_UT_hd2 = self.h3_UT_hd2(fea)
        fea= torch.cat((h1_PT_hd2,fea_encoder[1],hd3_UT_hd2),1)
        # fea=torch.cat((h1_PT_hd2,fea_encoder[1]),1)
        fea=self.h3_UT_hd2_conv(fea)
        fea=self.MSAB(fea)
        fea=self.hd2_UT_hd1(fea)
        h3_UT_hd1=self.hd3_UT_hd1(out)
        fea=torch.cat((fea,fea_encoder[0],h3_UT_hd1),1)
        # fea=torch.cat((fea,fea_encoder[0]),1)
        fea=self.hd3_UT_hd1_conv(fea)
        fea=self.MSAB1(fea)
        out = self.mapping(fea) + x
        return out


class FFTDeblur(nn.Module):
    def __init__(self,dim=31,outdim=31):
        super(FFTDeblur, self).__init__()
        self.fft_res = nn.Sequential(nn.Conv2d(dim, dim, 1),nn.Conv2d(dim,outdim,1))
        # self.fft_res = nn.Sequential(nn.Conv2d(dim, dim, 1),nn.Conv2d(dim,dim,1), nn.Conv2d(dim,dim,1),nn.Conv2d(dim, outdim, 1))
        # self.fft_res2=nn.Conv2d(dim,outdim,1)
    def forward(self,x):
        _,_,H,W=x.shape
        y = torch.fft.rfft2(x)
        y_imag = y.imag
        y_real = y.real
        # y_imag=self.fft_res(y_imag)
        # y_real=self.fft_res(y_real)
        y_imag=self.fft_res(y_imag)
        y_real=self.fft_res(y_real)
        # y_imag=self.fft_res(y_imag)+self.fft_res2(y_imag)
        # y_real=self.fft_res(y_real)+self.fft_res2(y_real)
        # 把实部，虚部组合起来。
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W))
        return y

# def seed_torch(seed=1029):
#     # System.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# channels=torch.nn.Conv2d(3)




class FFM(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=1):
        super(FFM, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.fca1 = FFTDeblur(self.dim, self.dim)
        self.downsize1 = nn.Conv2d(self.dim, 2*self.dim, 4, 2, 1, bias=False)

        self.fca2 = FFTDeblur(2*self.dim, 2*self.dim)
        self.downsize2 = nn.Conv2d(2*self.dim, 4*self.dim, 4, 2, 1, bias=False)

        self.fca3 = FFTDeblur(4*self.dim,4*self.dim)

        self.upSample2 = nn.ConvTranspose2d(4*self.dim, 2*self.dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upConv2 = nn.Conv2d(4*self.dim, 2*self.dim, 1, bias=False)

        self.upSample1 = nn.ConvTranspose2d(2*self.dim, self.dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upConv1 = nn.Conv2d(2*self.dim, self.dim, 1, bias=False)
        self.mapping = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)

        self.mapping = nn.Conv2d(self.dim,self.dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
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
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = self.embedding(x)  # [1,31,256,256]
        # Encoder
        # fea_encoder = []
        # for (MSAB,  FeaDownSample) in self.encoder_layers:
        #     fea = MSAB(fea)
        #     fea_encoder.append(fea)
        #     fea = FeaDownSample(fea)
        en1 = self.fca1(fea)
        en2 = self.downsize1(en1)
        en2 = self.fca2(en2)
        ##############################
        en3 = self.downsize2(en2)
        neck = self.fca3(en3)
        dn2_temp = self.upSample2(neck)
        dn2_temp1 = torch.cat((en2, dn2_temp), 1)
        dn2 = self.upConv2(dn2_temp1)
        dn1_temp = self.upSample1(dn2)
        dn1_temp1 = torch.cat((en1, dn1_temp), 1)
        dn1 = self.upConv1(dn1_temp1)
        out = self.mapping(dn1) + x

        return out
########################################
class FFMlite(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=1):
        super(FFMlite, self).__init__()
        self.dim = dim
        self.stage = stage
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.fca1 = FFTDeblur(self.dim, self.dim)
        self.downsize1 = nn.Conv2d(self.dim, 2*self.dim, 4, 2, 1, bias=False)

        self.fca2 = FFTDeblur(2*self.dim, 2*self.dim)

        self.upSample1 = nn.ConvTranspose2d(2*self.dim, self.dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upConv1 = nn.Conv2d(2*self.dim, self.dim, 1, bias=False)
        self.mapping = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)

        self.mapping = nn.Conv2d(self.dim,self.dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
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
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = self.embedding(x)  # [1,31,256,256]
        # Encoder
        # fea_encoder = []
        # for (MSAB,  FeaDownSample) in self.encoder_layers:
        #     fea = MSAB(fea)
        #     fea_encoder.append(fea)
        #     fea = FeaDownSample(fea)
        en1 = self.fca1(fea)
        en2 = self.downsize1(en1)
        bottneck2 = self.fca2(en2)
        ##############################
        dn1_temp = self.upSample1(bottneck2)
        dn1_temp1 = torch.cat((en1, dn1_temp), 1)
        dn1 = self.upConv1(dn1_temp1)
        out = self.mapping(dn1) + x

        return out
# ##############
# model=FFMlite()
# a=torch.randn(1,31,128,128)
# b=model(a)
# print()


class Our_2mst(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(Our_2mst, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        modules_body = [MST11(dim=31, stage=2, num_blocks=[1,1,4]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.FFt=FFTDeblur()
        # self.FFT=FFM()
        # self.FFT=FFMlite()

        # self.DWT=DWFTDeblur(31)
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h1= self.conv_out(h)
        h2=self.FFt(x)
        # h2=self.DWT(h)
        h=h1+h2+x
        return h[:, :, :h_inp, :w_inp]

########################################
# class Our_22mst(nn.Module):
#     def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
#         super(Our_22mst, self).__init__()
#         self.stage = stage
#         self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
#         modules_body = [MST11(dim=31, stage=2, num_blocks=[1,1,2]) for _ in range(stage)]
#         self.body = nn.Sequential(*modules_body)
#         self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
#
#     def forward(self, x):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         b, c, h_inp, w_inp = x.shape
#         hb, wb = 8, 8
#         pad_h = (hb - h_inp % hb) % hb
#         pad_w = (wb - w_inp % wb) % wb
#         x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
#         x = self.conv_in(x)
#         h = self.body(x)
#         h= self.conv_out(h)
#         h+=x
#
#         return h[:, :, :h_inp, :w_inp]
#
#
#

############################################3

# a=torch.randn(1,124,64,64)
# b=model(a)
# print(b.size())
#
# # # #
# model=Our_2mst()
# # model=FFM()
# a=torch.randn(1,3,256,256)
#
# # b=model(a)
# from thop import profile
# flops, params = profile(model, inputs=(a,))
# total = sum([param.nelement() for param in model.parameters()])
# print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
# print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
# print("Number of parameters: %.2fM" % (total / 1e6))
