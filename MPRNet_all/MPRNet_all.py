import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import torch.nn.init as init
import torchvision.utils

from pytorch_wavelets import DWTForward, DWTInverse
from MPRNet_all.ConvLSTM import ConvLSTM
import os


#############################
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, norm=None, bias=True, last_bias=0):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        if last_bias != 0:
            init.constant(self.conv2d.weight, 0)
            init.constant(self.conv2d.bias, last_bias)

    def forward(self, x):
        out = self.conv2d(x)

        return out


class ResidualBlock(nn.Module):

    def __init__(self, channels, groups=1, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = out + input

        return out


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## 
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## 
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
##
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, idwt_flag=False):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.idwt=DWTInverse(mode='symmetric', wave='haar')
        self.idwt_flag=idwt_flag

    def forward(self, x, x_img):
        x_new = x 
        x1 = self.conv1(x_new)

        x_ = self.conv2(x)

        if self.idwt_flag:
            B, C, H, W = x_.size()
            LL = x_[:, :, 0:int(H / 2), 0:int(W / 2)]
            HL = x_[:, :, 0:int(H / 2), int(W / 2):W]
            LH = x_[:, :, int(H / 2):H, 0:int(W / 2)]
            HH = x_[:, :, int(H / 2):H, int(W / 2):W]

            HL = HL.unsqueeze(2)
            LH = HL.unsqueeze(2)
            HH = HL.unsqueeze(2)
            h = torch.cat([HL, LH, HH], 2)
            H_ = []
            H_.append(h_)
            x_ = self.idwt((LL, H_))
        img = x_ + x_img  
        x2 = torch.sigmoid(self.conv3(img)) 

        x1 = x1 * x2
        x1 = x1 + x_new
        del x2
        del x_new
        del x_
        return x1, img


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)  
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)  
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, blocks, norm):
        super(Decoder, self).__init__()

        self.blocks = blocks
        self.norm = norm

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(n_feat + 2 * scale_unetfeats, bias=True, norm=self.norm))
        self.convlstm = ConvLSTM(input_size=n_feat + 2 * scale_unetfeats, hidden_size=n_feat + 2 * scale_unetfeats,
                                 kernel_size=3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

        self.convlstm = ConvLSTM(input_size=n_feat + 2 * scale_unetfeats, hidden_size=n_feat + 2 * scale_unetfeats,
                                 kernel_size=3)

    def forward(self, outs):
        enc1, enc2, enc3 = outs

        prev_state = None
        for b in range(self.blocks):
            RB = self.ResBlocks[b](enc3)
        state = self.convlstm(RB, prev_state)
        d3 = RB + state[0]

        dec3 = self.decoder_level3(d3) 

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x) 
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class MPRNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False, batch_size=2, N_frames=5, gpus=1, blocks=5, norm='IN'):
        super(MPRNet, self).__init__()

        act = nn.PReLU()

        self.dwt = DWTForward(J=1, mode='symmetric', wave='haar')
        self.idwt = DWTInverse(mode='symmetric', wave='haar')

        self.shallow_feat1ltop = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                               CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat1lbot = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                               CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat1rtop = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                               CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat1rbot = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                               CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.shallow_feat2left = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                               CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2right = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                                CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                                CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.conv3 = conv(out_c, out_c, kernel_size, bias=bias)

        self.changeSize1 = nn.Conv2d(int(batch_size * (N_frames + 1) / 2) // gpus, int(batch_size) // gpus, 3, 1, 1)
        self.changeSize2 = nn.Conv2d(int(batch_size * 3) // gpus, batch_size // gpus, 3, 1, 1)

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoderltop = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_encoderlbot = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_encoderrtop = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_encoderrbot = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)

        self.stage1_decoderleft = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, blocks, norm)
        self.stage1_decoderright = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, blocks, norm)

        self.stage2_encoderleft = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage2_encoderright = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)

        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, blocks, norm)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                    num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, idwt_flag=False)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, idwt_flag=False)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail1 = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)
        self.tail2 = conv(out_c, out_c, kernel_size, bias=bias)

        self.conv_gtLL = nn.Conv2d(3, 3, kernel_size=5, padding=0)
        self.conv_gtHL = nn.Conv2d(3, 3, kernel_size=5, padding=0)
        self.conv_gtLH = nn.Conv2d(3, 3, kernel_size=5, padding=0)
        self.conv_gtHH = nn.Conv2d(3, 3, kernel_size=5, padding=0)

    def forward(self, pcd_fea, pray_fea, gt):

        B, N, C, H1, W1 = pray_fea.size()
        B, N, C, H2, W2 = pcd_fea.size()

        #第三张图像
        x3 = pcd_fea[:, N//2, :, :, :]

        #
        x2_left = pcd_fea[:, 1:int(N / 2 + 1), :, :, :].contiguous()
        x2_left = x2_left.view(-1, C, H2, W2).contiguous()
        x2_left = x2_left[:, :, :, 0:int(W2 / 2)].contiguous()

        x2_right = pcd_fea[:, int(N / 2):int(N / 2 + 2), :, :, :].contiguous()
        x2_right = x2_right.view(-1, C, H2, W2).contiguous()
        x2_right = x2_right[:, :, :, int(W2 / 2):W2].contiguous()


        x1_top = pray_fea[:, 0:int(N / 2 + 1), :, :, :].contiguous()
        x1_top = x1_top.view(-1, C, H1, W1).contiguous()

        x1_ltop = x1_top[:, :, 0:int(H1 / 2), 0:int(W1 / 2)].contiguous()
        x1_rtop = x1_top[:, :, 0:int(H1 / 2), int(W1 / 2):W1].contiguous()

        x1_bot = pray_fea[:, int(N / 2):N, :, :, :].contiguous()
        x1_bot = x1_bot.view(-1, C, H1, W1).contiguous()

        x1_lbot = x1_bot[:, :, int(H1 / 2):H1, 0:int(W1 / 2)].contiguous()
        x1_rbot = x1_bot[:, :, int(H1 / 2):H1, int(W1 / 2):W1].contiguous()


        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------

        feat1_ltop = self.stage1_encoderltop(x1_ltop)
        feat1_rtop = self.stage1_encoderrtop(x1_rtop)
        feat1_lbot = self.stage1_encoderlbot(x1_lbot)
        feat1_rbot = self.stage1_encoderrbot(x1_rbot)

        feat1_left = [torch.cat((k, v), 2) for k, v in zip(feat1_ltop, feat1_lbot)]
        feat1_right = [torch.cat((k, v), 2) for k, v in zip(feat1_rtop, feat1_rbot)]
        res1_left = self.stage1_decoderleft(feat1_left)
        res1_right = self.stage1_decoderright(feat1_right)

        res1_left0 = self.changeSize1(res1_left[0].transpose(1, 0)).transpose(1, 0)
        res1_right0 = self.changeSize1(res1_right[0].transpose(1, 0)).transpose(1, 0)

        temp = torch.cat([res1_left0, res1_right0], 3)
        LL = temp[:, :, 0:int(H1 / 2), 0:int(W1 /2)]
        HL = temp[:, :, 0:int(H1 / 2), int(W1 / 2):W1]
        LH = temp[:, :, int(H1 / 2):H1, 0:int(W1 / 2)]
        HH = temp[:, :, int(H1 / 2):H1, int(W1 / 2):W1]

        HL = HL.unsqueeze(2)
        LH = LH.unsqueeze(2)
        HH = HH.unsqueeze(2)
        h_ = torch.cat([HL, LH, HH], 2)
        H_ = []
        H_.append(h_)
        temp = self.idwt((LL, H_))
        res1_left0 = temp[:,:,:,0:int(W2 / 2)]
        res1_right0 = temp[:,:,:,int(W2/2) : W2]
        del temp
        del LL
        del HL
        del LH
        del HH
        del feat1_rbot
        del feat1_lbot
        del feat1_rtop
        del feat1_ltop
        x2left_samfeats, stage1_img_left = self.sam12(res1_left0,gt[:, :, :, 0:int(W2 / 2)])
        x2right_samfeats, stage1_img_right = self.sam12(res1_right0, gt[:, :, :, int(W2 / 2):W2])

        stage1_img = torch.cat([stage1_img_left, stage1_img_right], 3)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------

        x2left_cat = torch.cat([x2_left, x2left_samfeats], 0)
        x2right_cat = torch.cat([x2_right, x2right_samfeats], 0)

        feat2_left = self.stage2_encoderleft(x2left_cat)
        feat2_right = self.stage2_encoderright(x2right_cat)
        del res1_left0
        del res1_right0
        del x2_left
        del x2_right
        del x2left_cat
        del x2right_cat
        del x1_bot
        del x1_top
        del x1_rtop
        del x1_rbot
        del x1_lbot
        del x1_ltop
        del res1_right
        del res1_left
        feat2 = [torch.cat((k, v), 3) for k, v in zip(feat2_left, feat2_right)]
        del feat2_left
        del feat2_right
        res2 = self.stage2_decoder(feat2) 
        res20 = self.changeSize2(res2[0].transpose(1, 0)).transpose(1, 0) 

        x3_samfeats, stage2_img = self.sam23(res20, gt) 

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------

        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1)) 
        del x3_samfeats
        feat2[0] = self.changeSize2(feat2[0].transpose(1, 0)).transpose(1, 0)
        feat2[1] = self.changeSize2(feat2[1].transpose(1, 0)).transpose(1, 0)
        feat2[2] = self.changeSize2(feat2[2].transpose(1, 0)).transpose(1, 0)
        res2[0] = self.changeSize2(res2[0].transpose(1, 0)).transpose(1, 0)
        res2[1] = self.changeSize2(res2[1].transpose(1, 0)).transpose(1, 0)
        res2[2] = self.changeSize2(res2[2].transpose(1, 0)).transpose(1, 0)

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail1(x3_cat)
  
        return [stage3_img + gt, stage2_img, stage1_img]
