### python lib
import os, sys, random, math, cv2, pickle, subprocess
import numpy as np
from PIL import Image
from natsort import natsorted
from glob import glob
from collections import OrderedDict
from math import log10
# from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim

### torch lib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

FLO_TAG = 202021.25
EPS = 1e-12

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


######################################################################################
##  Training utility
######################################################################################

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_last_path(path, session):
	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
	return x

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])


def myPSNR(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for
                          ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for
                 ind in range(len(pred_image_list))]

    return ssim_list

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps


def normalize_ImageNet_stats(batch):
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = (batch - mean) / std

    return batch_out


def img2tensor(img):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t


def tensor2img(img_t):
    img = img_t.detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def save_model(multi_model_res, multi_model_haze, multi_model_s2, optimizer, opts):
    # save opts
    opts_filename = os.path.join(opts.model_dir, "opts.pth")
    print("Save %s" % opts_filename)
    with open(opts_filename, 'wb') as f:
        pickle.dump(opts, f)

    # serialize model and optimizer to dict
    state_dict = {
        'multi_model_res': multi_model_res.state_dict(),
        'multi_model_haze': multi_model_haze.state_dict(),
        'multi_model_s2': multi_model_s2.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" % multi_model_res.epoch)
    print("Save %s" % model_filename)
    torch.save(state_dict, model_filename)


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def create_data_loader(data_set, opts, mode):
    if mode == 'train':
        total_samples = opts.OPTIM.NUM_EPOCHS * opts.OPTIM.BATCH_SIZE
    else:
        total_samples = opts.OPTIM.NUM_EPOCHS // 5 * opts.OPTIM.BATCH_SIZE

    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    data_loader = DataLoader(dataset=data_set, num_workers=opts.THREADS, batch_size=opts.OPTIM.BATCH_SIZE, sampler=sampler,
                             pin_memory=True)

    return data_loader


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])

    return N


######################################################################################
##  Image utility
######################################################################################


def rotate_image(img, degree, interp=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, degree, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    img_out = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), flags=interp + cv2.WARP_FILL_OUTLIERS)

    return img_out


def numpy_to_PIL(img_np):
    img_PIL = np.uint8(img_np * 255)
    img_PIL = Image.fromarray(img_PIL)

    return img_PIL


def PIL_to_numpy(img_PIL):
    img_np = np.asarray(img_PIL)
    img_np = np.float32(img_np) / 255.0

    return img_np


def read_img(filename, grayscale=0):
    if grayscale: 
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" % filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)

        if img is None:
            raise Exception("Image %s does not exist" % filename)

        img = img[:, :, ::-1]

    img = np.float32(img) / 255.0

    return img


def save_img(img, filename):
    if img.ndim == 3:
        img = img[:, :, ::-1] 
    img = np.uint8(img * 255.0)

    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])



def read_flo(filename):
    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)

        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' % filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))

            data = np.fromfile(f, np.float32, count=2 * w * h)

            flow = np.resize(data, (h, w, 2))

    return flow


def save_flo(flow, filename):
    with open(filename, 'wb') as f:
        tag = np.array([FLO_TAG], dtype=np.float32)

        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        tag.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)


def resize_flow(flow, W_out=0, H_out=0, scale=0):
    if W_out == 0 and H_out == 0 and scale == 0:
        raise Exception("(W_out, H_out) or scale should be non-zero")

    H_in = flow.shape[0]
    W_in = flow.shape[1]

    if scale == 0:
        y_scale = float(H_out) / H_in
        x_scale = float(W_out) / W_in
    else:
        y_scale = scale
        x_scale = scale

    flow_out = cv2.resize(flow, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)

    flow_out[:, :, 0] = flow_out[:, :, 0] * x_scale
    flow_out[:, :, 1] = flow_out[:, :, 1] * y_scale

    return flow_out


def rotate_flow(flow, degree, interp=cv2.INTER_LINEAR):
    ## angle in radian
    angle = math.radians(degree)

    H = flow.shape[0]
    W = flow.shape[1]
    flow_out = rotate_image(flow, degree, interp)

    fu = flow_out[:, :, 0] * math.cos(-angle) - flow_out[:, :, 1] * math.sin(-angle)
    fv = flow_out[:, :, 0] * math.sin(-angle) + flow_out[:, :, 1] * math.cos(-angle)

    flow_out[:, :, 0] = fu
    flow_out[:, :, 1] = fv

    return flow_out


def hflip_flow(flow):
    flow_out = cv2.flip(flow, flipCode=0)
    flow_out[:, :, 0] = flow_out[:, :, 0] * (-1)

    return flow_out


def vflip_flow(flow):
    flow_out = cv2.flip(flow, flipCode=1)
    flow_out[:, :, 1] = flow_out[:, :, 1] * (-1)

    return flow_out


def flow_to_rgb(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    # print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.float32(img) / 255.0


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_flow_magnitude(flow):
    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag


def compute_flow_gradients(flow):
    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


######################################################################################
##  Other utility
######################################################################################

def save_vector_to_txt(matrix, filename):
    with open(filename, 'w') as f:
        print("Save %s" % filename)

        for i in range(matrix.size):
            line = "%f" % matrix[i]
            f.write("%s\n" % line)


def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True)


def make_video(input_dir, img_fmt, video_filename, fps=24):
    cmd = "ffmpeg -y -loglevel error -framerate %s -i %s/%s -vcodec libx264 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" %s" \
          % (fps, input_dir, img_fmt, video_filename)

    run_cmd(cmd)
