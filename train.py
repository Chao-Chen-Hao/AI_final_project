import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
from torch.utils import data, model_zoo
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import time
import yaml
from tensorboardX import SummaryWriter

from trainer import Trainer
from utils.tool import adjust_learning_rate, Timer 
from utils.evaluate import evaluate
from dataset import QDDataSet

#### Hyperparameters ####
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
POWER = 0.9
RANDOM_SEED = 1234

#### Model Settings  ####
BATCH_SIZE = 10
NUM_STEPS = 100001
SAVE_PRED_EVERY = 1000

####  Path Settings  ####
DATA_DIRECTORY = '../../Warehouse/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
SNAPSHOT_DIR = './snapshots'
LOG_DIR = './log'
RESTORE_FROM = '../../weights/weights/gta5/deeplabv3+/source/drn/model_34.80.pth'
GT_DIR = '../../Warehouse/Cityscapes/data/gtFine/val'
GT_LIST_PATH = './dataset/cityscapes_list'
RESULT_DIR = './result/gta5'

def get_arguments():
    parser = argparse.ArgumentParser(description="Network")
    #### Hyperparameters ####
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
                        help="DropRate.")

    #### Model Settings  ####
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")

    ####  Path Settings  ####
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--gt-dir", type=str, default=GT_DIR,
                        help="Path to the folder containing the gt data.")
    parser.add_argument("--gt-list", type=str, default=GT_LIST_PATH,
                        help="Path to the folder includeing the list of gt.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Path to the results. (prediction)")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    
    return parser.parse_args()


args = get_arguments()

# save opts
if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

def main():

    cudnn.enabled = True
    cudnn.benchmark = True

    trainloader = data.DataLoader(
        QDDataSet(args.data_dir, args.data_list,
                max_iters = args.num_steps * args.batch_size,
                resize_size = (72, 72),
                crop_size = (64, 64),
                mirror=args.random_mirror),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    trainloader_iter = enumerate(trainloader)
    
    # set up tensor board
    args.log_dir += '/'+ os.path.basename(args.snapshot_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)
    
    best_accuracy = 0

    for i_iter in range(args.num_steps):
        loss_value = 0

        adjust_learning_rate(Trainer.gen_opt , i_iter, args)

        _, batch = trainloader_iter.__next__()
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with Timer("Elapsed time in update: %f"):
            loss = Trainer.gen_update(images, labels, i_iter)
            loss_value += loss.item()

        if args.tensorboard:
            scalar_info = {
                'loss': loss_value
            }

            if i_iter % 100 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('\033[1m iter = %8d/%8d \033[0m loss = %.3f' %(i_iter, args.num_steps, loss_value))

        del loss

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            accuracy = evaluate(Trainer.model)
            writer.add_scalar('accuracy', accuracy, int(i_iter/args.save_pred_every)) # (TB)
            if accuracy > best_accuracy:
                print('save model ...')
                best_accuracy = accuracy
                torch.save(Trainer.model.state_dict(), osp.join(args.snapshot_dir, 'model.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
