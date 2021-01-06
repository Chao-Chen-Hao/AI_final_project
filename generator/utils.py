import os
import os.path as osp
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
import csv

def get_labels(file_name='../Warehouse/dataset/demo.csv'):
    # Read the demo data and convert to a label list.
    # OUTPUT: a list of the idx of the labels to be generated.
    class_list = {'airplane': 0, 'bee': 1, 'bicycle': 2, 'bird': 3, 'butterfly': 4, 'cake': 5,
            'camera': 6, 'cat': 7, 'chair': 8, 'clock': 9, 'computer': 10,
            'diamond': 11, 'door': 12, 'ear': 13, 'guitar': 14, 'hamburger': 15,
            'hammer': 16, 'hand': 17, 'hat': 18, 'ladder': 19, 'leaf': 20,
            'lion': 21, 'pencil': 22, 'rabbit': 23, 'scissors': 24, 'shoe': 25,
            'star': 26, 'sword': 27, 'The Eiffel Tower': 28, 'tree': 29}

    datafile = open(file_name, 'r')
    datareader = csv.reader(datafile, delimiter=',')
    output_list = []
    idx = 0
    for data in datareader:
        idx = idx + 1
        if(idx == 1): continue

        index = data[0]
        stroke = data[1]
        label = data[2]

        output_list.append(class_list[label])

    return output_list

def generate_noise(label, id, noise_size=500, seed=123, gen=0, batch=1):
    # Generate an input noise vector.
    # OUTPUT: noise vector with the size (b, 500, 1, 1).
    if (gen == 0):
        input_noise = np.zeros((batch, noise_size, 1, 1), np.float)
        input_noise[0, label * 10 + id + seed, 0, 0] = 1
        input_noise = torch.tensor(input_noise)
    else:
        input_noise = torch.tensor(torch.randn(batch, noise_size, 1 , 1))
    return input_noise