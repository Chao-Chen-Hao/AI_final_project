import argparse
import torch
import numpy as np
from torch.utils import data, model_zoo
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp

from dataset import QDDataSet
from efficientnet_pytorch import EfficientNet

####  Test Settings  ####
TOTAL = 200
BATCH_SIZE = 200

####  Path Settings  ####
DATA_DIRECTORY = '../Warehouse/dataset/test'
RESTORE_FROM = 'snapshots/model.pth'
INPUT_FILE = '../Warehouse/dataset/testing.csv'

def get_arguments():
    parser = argparse.ArgumentParser(description="Network")
    #### Model Settings  ####
    parser.add_argument("--total", type=int, default=TOTAL,
                        help="Number of testing examples.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    ####  Path Settings  ####
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--input-file", type=str, default=INPUT_FILE,
                        help="The input file.")
    
    return parser.parse_args()

def test(args, model):

    results = []
    model.eval()
    
    testloader = data.DataLoader(
        QDDataSet(root=args.data_dir, t_set='test', max_iters=args.total/args.batch_size,
                resize_size=(128, 128), crop_size=(128, 128), mirror=False),
    batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    sm = torch.nn.Softmax(dim = 1)

    for index, data_img in enumerate(testloader):
        image = data_img
        inputs = image.cuda()
        
        print('\r>>>>Extracting feature...%03d/%03d'%(index*args.batch_size, args.total))

        with torch.no_grad():
            output_batch = sm(model(inputs))
            output_batch = output_batch.cpu().data.numpy()
            del inputs

        output_batch = np.asarray(np.argmax(output_batch, axis=1), dtype=np.uint8)
        results = list(output_batch.flatten())
        print(results)

    f = open(args.data_dir + "/testing.txt", "a+")
    for label in results:
        f.write(str(label) + '\n')
    f.close()


def main():
    args = get_arguments()

    cudnn.enabled = True
    cudnn.benchmark = True

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=30)
    model = model.cuda()
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    
    test(args, model)
    print("FNIISH.")

if __name__ == '__main__':
    main()
