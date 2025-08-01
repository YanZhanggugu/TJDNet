import os, sys, math, random, glob, cv2
import numpy as np

import torch
import torch.utils.data as data

import utils.util as utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]


class MultiFramesDataset(data.Dataset):

    def __init__(self, opts, dataset, mode):
        super(MultiFramesDataset, self).__init__()

        self.opts = opts
        self.mode = mode
        self.task_videos = []
        self.num_frames = []
        self.dataset_task_list = []

        list_filename = os.path.join(opts.DATASET.LIST_DIR, "%s_%s.txt" % (dataset, mode))
        print(list_filename)

        with open(list_filename) as f:
            videos = [line.rstrip() for line in f.readlines()]

        for video in videos:
            self.task_videos.append([os.path.join(video)])

            input_dir = os.path.join(self.opts.DATASET.DATA_DIR, self.mode, video, "GT")
            frame_list = glob.glob(os.path.join(input_dir, '*.jpg'))  

            if len(frame_list) == 0:
                raise Exception("No frames in %s" % input_dir)

            self.num_frames.append(len(frame_list))

        print(
            "[%s] Total %d videos (%d frames)" % (self.__class__.__name__, len(self.task_videos), sum(self.num_frames)))

    def __len__(self):
        return len(self.task_videos)  

    def __getitem__(self, index):

        N = self.num_frames[index] 
        T = random.randint(0, N - self.opts.DATASET.SAMPLE_FRAMES) 

        video = self.task_videos[index][0]

        input_dir = os.path.join(self.opts.DATASET.DATA_DIR, self.mode, video, "GT") 
        process_dir = os.path.join(self.opts.DATASET.DATA_DIR, self.mode, video, "RAIN") 

        frame_i = []
        frame_p = []

        for t in range(T + 1, T + self.opts.DATASET.SAMPLE_FRAMES + 1):
            frame_p.append(utils.read_img(os.path.join(process_dir, "%d.jpg" % t)))
        mid_frame = T + 1 + self.opts.DATASET.SAMPLE_FRAMES / 2
        frame_i.append(utils.read_img(os.path.join(input_dir, "%d.jpg" % mid_frame)))

        if self.mode == 'train' or self.mode == 'train_test': 
            if self.opts.DATASET.GEOMETRY_AUG:

                H_in = frame_i[0].shape[0]
                W_in = frame_i[0].shape[1] 

                sc = np.random.uniform(self.opts.DATASET.SCALE_MIN, self.opts.DATASET.SCALE_MAX)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                if H_out < W_out:
                    if H_out < self.opts.DATASET.CROP_SIZE:
                        H_out = self.opts.DATASET.CROP_SIZE
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else:  
                    if W_out < self.opts.DATASET.CROP_SIZE:
                        W_out = self.opts.DATASET.CROP_SIZE
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.opts.DATASET.SAMPLE_FRAMES):
                    frame_p[t] = cv2.resize(frame_p[t], (W_out, H_out))
                frame_i[0] = cv2.resize(frame_i[0], (W_out, H_out))

            cropper = RandomCrop(frame_i[0].shape[:2], (self.opts.DATASET.CROP_SIZE, self.opts.DATASET.CROP_SIZE))

            for t in range(self.opts.DATASET.SAMPLE_FRAMES):
                frame_p[t] = cropper(frame_p[t])
            frame_i[0] = cropper(frame_i[0])

            if self.opts.DATASET.GEOMETRY_AUG:
                if np.random.random() >= 0.5:
                    for t in range(self.opts.DATASET.SAMPLE_FRAMES):
                        frame_p[t] = cv2.flip(frame_p[t], flipCode=0)
                    frame_i[0] = cv2.flip(frame_i[0], flipCode=0)

            if self.opts.DATASET.ORDER_AUG:
                if np.random.random() >= 0.5:
                    frame_p.reverse()

        elif self.mode == "test": 
            H_i = frame_i[0].shape[0]
            W_i = frame_i[0].shape[1]

            H_o = int(math.ceil(float(H_i) / self.opts.size_multiplier) * self.opts.size_multiplier)
            W_o = int(math.ceil(float(W_i) / self.opts.size_multiplier) * self.opts.size_multiplier)

            for t in range(self.opts.DATASET.SAMPLE_FRAMES):
                frame_i[t] = cv2.resize(frame_i[t], (W_o, H_o))
                frame_p[t] = cv2.resize(frame_p[t], (W_o, H_o))
        else:
            raise Exception("Unknown mode (%s)" % self.mode)


        gt_frames = torch.from_numpy(frame_i[0].transpose(2, 0, 1).astype(np.float32)).contiguous()
        rain_frames = []
        for t in range(self.opts.DATASET.SAMPLE_FRAMES):
            rain_frames.append(torch.from_numpy(frame_p[t].transpose(2, 0, 1).astype(np.float32)).contiguous())

        return [gt_frames, rain_frames]
