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
from tensorboardX import SummaryWriter

from trainer import QDTrainer
from tool import adjust_learning_rate, Timer 
from dataset import QDDataSet

#### Hyperparameters ####
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 5e-1
MOMENTUM = 0.9
POWER = 0.9
RANDOM_SEED = 1234

#### Model Settings  ####
BATCH_SIZE = 200
NUM_STEPS = 1000001
SAVE_PRED_EVERY = 300

####  Path Settings  ####
DATA_DIRECTORY = '../Warehouse/dataset/train/'
DATA_VAL_DIRECTORY = '../Warehouse/dataset/val/'
SNAPSHOT_DIR = './snapshots'
LOG_DIR = './log'
RESTORE_FROM = None

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
                        
    #### Model Settings  ####
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")

    ####  Path Settings  ####
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-val-dir", type=str, default=DATA_VAL_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    
    return parser.parse_args()

def evaluate(args, model):
    
    model.eval()

    total = 15000
    testloader = data.DataLoader(
        QDDataSet(root=args.data_val_dir, t_set='val', max_iters=total/args.batch_size,
                resize_size=(128, 128), crop_size=(128, 128), mirror=False),
    batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    testloader_iter = enumerate(testloader)

    sm = torch.nn.Softmax(dim = 1)
    accuracy = 0

    for index, data_img in enumerate(testloader):
        image, label = data_img
        inputs = image.cuda()
        
        print('\r>>>>Extracting feature...%03d/%03d'%(index*args.batch_size, total))

        with torch.no_grad():
            output_batch = sm(model(inputs))
            output_batch = output_batch.cpu().data.numpy()
            del inputs

        output_batch = np.asarray(np.argmax(output_batch, axis=1), dtype=np.uint8)
        num_match = np.sum(output_batch == np.array(label))
        print(num_match)
        accuracy = accuracy + num_match
    
    accuracy = accuracy / total
    print(accuracy)

    model.train()
    
    return accuracy

def main():
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    writer = SummaryWriter(args.log_dir)
    Trainer = QDTrainer(args)

    cudnn.enabled = True
    cudnn.benchmark = True

    trainloader = data.DataLoader(
        QDDataSet(root=args.data_dir, max_iters=args.num_steps * args.batch_size,
                resize_size=(128, 128), crop_size=(100, 100), mirror=True),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    trainloader_iter = enumerate(trainloader)
    
    best_accuracy = 0

    for i_iter in range(args.num_steps):
        loss_value = 0

        adjust_learning_rate(Trainer.gen_opt, i_iter, args)

        _, batch = trainloader_iter.__next__()
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with Timer("Elapsed time in update: %f"):
            loss = Trainer.gen_update(images, labels, i_iter)
            loss_value += loss.item()

        scalar_info = {'loss': loss_value}
        if i_iter % 50 == 0:
            for key, val in scalar_info.items():
                writer.add_scalar(key, val, i_iter)

        print('\033[1m iter = %8d/%8d \033[0m loss = %.3f' %(i_iter, args.num_steps, loss_value))

        del loss

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            accuracy = evaluate(args, Trainer.model)
            writer.add_scalar('accuracy', accuracy, int(i_iter/args.save_pred_every)) # (TB)
            if accuracy > best_accuracy:
                print('save model ...')
                best_accuracy = accuracy
                torch.save(Trainer.model.state_dict(), osp.join(args.snapshot_dir, 'model.pth'))

    writer.close()

if __name__ == '__main__':
    main()
