# python -m tensorboard.main --logdir 'tensorboard/ZWZNet' --load_fast true
# tensorboard --logdir=/home/disk1/zwz/derainCode1/tensorboard/ZWZNet/

from __future__ import print_function

import os
from config import Config
opts = Config('training.yml')
print(opts)

gpus = ','.join([str(i) for i in opts.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import sys, argparse, glob, re, math, copy, pickle, random, cv2
import time

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from data import datasets_multiple
from utils import util as utils

import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pytorch_wavelets import DWTForward

from ZWZ_Net import ZWZNet_all

import numpy as np

plt.switch_backend('agg')

def main():
    random.seed(opts.SEED)
    np.random.seed(opts.SEED)
    torch.manual_seed(opts.SEED)
    torch.cuda.manual_seed_all(opts.SEED)

    start_epoch = 1
    mode = opts.MODEL.MODE
    session = opts.MODEL.SESSION

    result_dir = os.path.join(opts.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir = os.path.join(opts.TRAINING.SAVE_DIR, mode, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    model_restoration = ZWZNet_all.ZWZNet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    new_lr = opts.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)


    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,after_scheduler=scheduler_cosine)

    if opts.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)


    train_dataset = datasets_multiple.MultiFramesDataset(opts, "rain_removal", "train")
    print(train_dataset)
    train_loader = utils.create_data_loader(train_dataset, opts, "train")

    val_dataset = datasets_multiple.MultiFramesDataset(opts, "rain_removal", "train_test")
    val_loader = utils.create_data_loader(val_dataset, opts, "train_test")

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opts.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0

    writer = SummaryWriter('./tensorboard/ZWZNet')
    for epoch in range(start_epoch, opts.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        loss_list = []
        psnr_train_list = []
        ssim_train_list = []

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            for param in model_restoration.parameters():
                param.grad = None

            target, input_ = data
            target = target.cuda()
            input_ = torch.stack(input_, dim=0).contiguous().cuda()
            input_ = torch.transpose(input_, 0, 1)
            restored_first, restored = model_restoration(input_)
            stage3 = restored_first[0]
            stage2 = restored_first[1]
            stage1 = restored_first[2]
            del(restored_first)

            loss = losses.caculate_loss(stage3, stage2, stage1, restored, target)
            del(stage1)
            del(stage2)
            del(stage3)

            loss_list.append(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            psnr_train_list.extend(utils.myPSNR(restored, target))
            ssim_train_list.extend(utils.to_ssim_skimage(restored, target))
            del(restored)
            del(target)

        train_psnr = sum(psnr_train_list) / len(psnr_train_list)
        train_ssim = sum(ssim_train_list) / len(ssim_train_list)
        if epoch_loss > 0:
            writer.add_scalar('loss', epoch_loss, global_step=epoch)
            writer.add_scalar('train_psnr', train_psnr, global_step=epoch)
            writer.add_scalar('train_ssim', train_ssim, global_step=epoch)
            writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=epoch)

        if epoch % opts.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_list = []
            ssim_val_list = []

            for ii, data_val in enumerate((val_loader), 0):
                target, input_ = data_val
                target = target.cuda()
                input_ = torch.stack(input_, dim=0).contiguous().cuda()
                input_ = torch.transpose(input_, 0, 1)

                with torch.no_grad():
                    restored_first, restored = model_restoration(input_)
                    del(restored_first)

                psnr_val_list.extend(utils.myPSNR(restored, target))
                ssim_val_list.extend(utils.to_ssim_skimage(restored, target))
                del(restored)
                del(target)

            val_psnr = sum(psnr_val_list) / len(psnr_val_list)
            val_ssim = sum(ssim_val_list) / len(ssim_val_list)

            if val_psnr > 0:
                writer.add_scalar('val_psnr', val_psnr, global_step=epoch // opts.TRAINING.VAL_AFTER_EVERY)
                writer.add_scalar('val_ssim', val_ssim, global_step=epoch // opts.TRAINING.VAL_AFTER_EVERY)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))

            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, val_psnr, best_epoch, best_psnr))

            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tPSNR: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  train_psnr, epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_best.pth"))

if __name__ == '__main__':
    main()