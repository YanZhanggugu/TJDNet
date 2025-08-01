from locale import normalize
import numpy as np
import os
import argparse
import glob
import math
import time

import torchvision.utils
from matplotlib import pyplot as plt
import cv2

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.nn as nn
import torch
torch.backends.cudnn.benchmark = True
from utils import util as utils
from skimage import img_as_ubyte

from ZWZ_Net import ZWZNet_all
from skimage import img_as_ubyte


plt.switch_backend('agg')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='Video Deraining using ZWZNet')

    parser.add_argument('-dataset', type=str, default="rain_removal", help='dataset to test')
    parser.add_argument('-phase', type=str, default="test", choices=["train", "test"])
    parser.add_argument('--input_dir', default='./datasets/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./datasets/', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./checkpoints/Deraining/models/model_epoch_500.pth', type=str,
                        help='Path to weights')
    parser.add_argument('-list_dir', default='./file_list/', type=str, help='path to list folder')
    parser.add_argument('-redo', action="store_true", help='Re-generate results')

    args = parser.parse_args()
    args.cuda = True

    args.size_multiplier = 8
    print(args)
    device_ids = [0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")

    model_restoration = ZWZNet_all.ZWZNet()


    utils.load_checkpoint(model_restoration, args.weights) 
    print("===>Testing using weights: ", args.weights)
    model_restoration.to(device)
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)
    model_restoration.train()
    print("Total number of model parameters: ", count_parameters(model_restoration))

    model_restoration.eval()

    list_filename = os.path.join(args.list_dir, "%s_%s_real.txt" % (args.dataset, args.phase)) 

    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]
    time_sum = 0
    for v in range(len(video_list)):
        video = video_list[v] 
        print("Test on %s-%s video %d/%d: %s" % (args.dataset, args.phase, v + 1, len(video_list), video))

        input_dir = os.path.join(args.input_dir, args.phase, video, "RAIN")
        output_dir = os.path.join(args.result_dir, args.phase, video, "result_best")

        print(r"Input dir: ", input_dir)
        print(r"Output dir", output_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        frame_list = glob.glob(os.path.join(input_dir, "*.jpg"))
        output_list = glob.glob(os.path.join(output_dir, "*.jpg"))

        if len(frame_list) == len(output_list):
            print("Output frames exist, skip...")
            continue

        start_time = time.time()
        for t in range(1, len(frame_list) - 3):
            frame_i1 = utils.read_img(os.path.join(input_dir, "%s.jpg" % str(t).zfill(5)))
            frame_i2 = utils.read_img(os.path.join(input_dir, "%s.jpg" % str(t + 1).zfill(5)))
            frame_i3 = utils.read_img(os.path.join(input_dir, "%s.jpg" % str(t + 2).zfill(5)))
            frame_i4 = utils.read_img(os.path.join(input_dir, "%s.jpg" % str(t + 3).zfill(5)))
            frame_i5 = utils.read_img(os.path.join(input_dir, "%s.jpg" % str(t + 4).zfill(5)))

            H_orig = frame_i1.shape[0]
            W_orig = frame_i1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / args.size_multiplier) * args.size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / args.size_multiplier) * args.size_multiplier)

            with torch.no_grad():
                frame_i1 = utils.img2tensor(frame_i1).to(device)
                frame_i2 = utils.img2tensor(frame_i2).to(device)
                frame_i3 = utils.img2tensor(frame_i3).to(device)
                frame_i4 = utils.img2tensor(frame_i4).to(device)
                frame_i5 = utils.img2tensor(frame_i5).to(device)

                [b, c, h, w] = frame_i1.shape
                frame_i1_new = torch.zeros(b, c, H_sc, W_sc).to(device)
                frame_i2_new = torch.zeros(b, c, H_sc, W_sc).to(device)
                frame_i3_new = torch.zeros(b, c, H_sc, W_sc).to(device)
                frame_i4_new = torch.zeros(b, c, H_sc, W_sc).to(device)
                frame_i5_new = torch.zeros(b, c, H_sc, W_sc).to(device)

                frame_i1_new[:, :, :h, :w] = frame_i1
                frame_i2_new[:, :, :h, :w] = frame_i2
                frame_i3_new[:, :, :h, :w] = frame_i3
                frame_i4_new[:, :, :h, :w] = frame_i4
                frame_i5_new[:, :, :h, :w] = frame_i5

                for s in range(h, H_sc):
                    frame_i1_new[:, :, s, :] = frame_i1_new[:, :, h - 1, :]
                    frame_i2_new[:, :, s, :] = frame_i2_new[:, :, h - 1, :]
                    frame_i3_new[:, :, s, :] = frame_i3_new[:, :, h - 1, :]
                    frame_i4_new[:, :, s, :] = frame_i4_new[:, :, h - 1, :]
                    frame_i5_new[:, :, s, :] = frame_i5_new[:, :, h - 1, :]

                for s in range(w, W_sc):
                    frame_i1_new[:, :, :, s] = frame_i1_new[:, :, :, w - 1]
                    frame_i2_new[:, :, :, s] = frame_i2_new[:, :, :, w - 1]
                    frame_i3_new[:, :, :, s] = frame_i3_new[:, :, :, w - 1]
                    frame_i4_new[:, :, :, s] = frame_i4_new[:, :, :, w - 1]
                    frame_i5_new[:, :, :, s] = frame_i5_new[:, :, :, w - 1]

                frame_i1 = frame_i1_new
                frame_i2 = frame_i2_new
                frame_i3 = frame_i3_new
                frame_i4 = frame_i4_new
                frame_i5 = frame_i5_new

                inputs = torch.stack((frame_i1, frame_i2, frame_i3, frame_i4, frame_i5), dim=0)

                inputs = torch.transpose(inputs, 0, 1)
                inputs = inputs.repeat([2, 1, 1, 1, 1]).to(device)

                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                restored_first, restored= model_restoration(inputs)
                #restored = restored[0][:, :, :h, :w]
                restored = restored[0][:, :h, :w]
                torchvision.utils.save_image(restored, (os.path.join(output_dir, str(t + 2) + '.png')))

        print(time.time()-start_time)
        time_sum += time.time()-start_time
    print("all_time:%d"%(time_sum))

if __name__ == '__main__':
    main()
