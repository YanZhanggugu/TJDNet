import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


def caculate_loss(restored, satge2_img, stage3_img, derain_res, target):
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()

    loss_char1 = criterion_char(restored, target)
    loss_edge1 = criterion_edge(restored, target)

    loss_char2 = criterion_char(satge2_img, target)
    loss_edge2 = criterion_edge(satge2_img, target)

    loss_char3 = criterion_char(stage3_img, target)
    loss_edge3 = criterion_edge(stage3_img, target)

    loss_char_derain = 2 * criterion_char(derain_res, target)
    loss_edge_derain = 2 * criterion_edge(derain_res, target)

    loss_char = loss_char1 + loss_char2 + loss_char3 + loss_char_derain
    loss_edge = loss_edge1 + loss_edge2 + loss_edge3 + loss_edge_derain

    loss = (loss_char) + (0.05 * loss_edge)

    return loss