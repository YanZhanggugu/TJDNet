from MPRNet_all import MPRNet_all
from Last_part import LastPart
from PCD_part import EDVR_arch
from pytorch_wavelets import DWTForward, DWTInverse

import torch, time
import torch.nn as nn


class ZWZNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, nf=40, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=2,
                 scale_unetfeats=20, scale_orsnetfeats=16, predeblur=False, HR_in=False, w_TSA=False,
                 kernel_size=3, num_cab=8, reduction=4, bias=False, batch_size=8, gpus=4, blocks=5, norm='IN'):
        super(ZWZNet, self).__init__()
        self.center = center
        self.PCD_LL = EDVR_arch.EDVR(nf=nf, nframes=nframes, groups=groups, front_RBs=front_RBs, back_RBs=back_RBs,
                                  center=center, predeblur=predeblur, HR_in=HR_in, w_TSA=w_TSA)
        self.PCD_LH = EDVR_arch.EDVR(nf=nf, nframes=nframes, groups=groups, front_RBs=front_RBs, back_RBs=back_RBs,
                                  center=center, predeblur=predeblur, HR_in=HR_in, w_TSA=w_TSA)
        self.PCD_HL = EDVR_arch.EDVR(nf=nf, nframes=nframes, groups=groups, front_RBs=front_RBs, back_RBs=back_RBs,
                                  center=center, predeblur=predeblur, HR_in=HR_in, w_TSA=w_TSA)
        self.PCD_other = EDVR_arch.EDVR(nf=nf, nframes=nframes, groups=groups, front_RBs=front_RBs, back_RBs=back_RBs,
                                  center=center, predeblur=predeblur, HR_in=HR_in, w_TSA=w_TSA)

        self.PCD = EDVR_arch.EDVR(nf=nf, nframes=nframes, groups=groups, front_RBs=front_RBs, back_RBs=back_RBs,
                                  center=center, predeblur=predeblur, HR_in=HR_in, w_TSA=w_TSA)

        self.MPR = MPRNet_all.MPRNet(in_c=in_c, out_c=out_c, n_feat=nf, scale_unetfeats=scale_unetfeats,
                                 scale_orsnetfeats=scale_orsnetfeats, num_cab=num_cab, kernel_size=kernel_size,
                                 reduction=reduction, bias=bias, batch_size=batch_size, N_frames=nframes, gpus=gpus,
                                 blocks=blocks, norm=norm)
        self.dwt = DWTForward(J=1, mode='symmetric', wave='haar')
        self.conv_LL = nn.Conv2d(3, 3, kernel_size=5, padding=0)
        self.conv_HL = nn.Conv2d(3, 3, kernel_size=5, padding=0)
        self.conv_LH = nn.Conv2d(3, 3, kernel_size=5, padding=0)
        self.conv_HH = nn.Conv2d(3, 3, kernel_size=5, padding=0)

        self.LastNet = LastPart.LastPart()

    def forward(self, inputs):
        inp_center = inputs[:, self.center, :, :, :].contiguous()
        B, N, C, H, W = inputs.size()
        pray_inputs = inputs.contiguous().view(-1, C, H, W)

        LL, H_ = self.dwt(pray_inputs)
        HL = H_[0][:, :, 0, :, :]
        LH = H_[0][:, :, 1, :, :] 
        HH = H_[0][:, :, 2, :, :] 

        LL = LL.view(B, N, -1, int(H // 2), int(W // 2))
        HL = HL.view(B, N, -1, int(H // 2), int(W // 2))
        LH = LH.view(B, N, -1, int(H // 2), int(W // 2))
        HH = HH.view(B, N, -1, int(H // 2), int(W // 2))

        LL_fea = self.PCD_LL(LL)
        HL_fea = self.PCD_other(HL)
        LH_fea = self.PCD_LH(LH)
        HH_fea = self.PCD_other(HH)

        top_cat = torch.cat([LL_fea, HL_fea], dim=4).contiguous()
        bottom_cat = torch.cat([LH_fea, HH_fea], dim=4).contiguous()
        pray_fea = torch.cat([top_cat, bottom_cat], dim=3).contiguous()
        del LL_fea
        del HL_fea
        del LH_fea
        del HH_fea
        del top_cat
        del bottom_cat
        pcd_fea = self.PCD(inputs)

        res = self.MPR(pcd_fea, pray_fea, inp_center)
        del(pcd_fea)
        del(pray_fea)

        derain = self.LastNet(res, inp_center)

        return res, derain