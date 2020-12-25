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

ImageFile.LOAD_TRUNCATED_IMAGES = True

class QDDataSet(data.Dataset):
    def __init__(self, root, t_set='train', max_iters=None, resize_size=(72, 72), crop_size=(64, 64), mirror=True ):
        self.root = root
        self.crop_size = crop_size
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.max_iters = max_iters
        self.h = crop_size[0]
        self.w = crop_size[1]
        if(t_set == 'train'):
            self.img_ids = [i_id.strip() for i_id in open(osp.join(self.root, "train_list.txt"))]
        if(t_set == 'val'):
            self.img_ids = [i_id.strip() for i_id in open(osp.join(self.root, "val_list.txt"))]
        self.files = []
        self.class_list = {'airplane': 0, 'bee': 1, 'bicycle': 2, 'bird': 3, 'butterfly': 4, 'cake': 5,
                            'camera': 6, 'cat': 7, 'chair': 8, 'clock': 9, 'computer': 10,
                            'diamond': 11, 'door': 12, 'ear': 13, 'guitar': 14, 'hamburger': 15,
                            'hammer': 16, 'hand': 17, 'hat': 18, 'ladder': 19, 'leaf': 20,
                            'lion': 21, 'pencil': 22, 'rabbit': 23, 'scissors': 24, 'shoe': 25,
                            'star': 26, 'sword': 27, 'The Eiffel Tower': 28, 'tree': 29}

        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            label = self.class_list[name.split('/')[0]]
            self.files.append({
                "img": img_file,
                "label": label
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = datafiles["label"]
        #label = np.asarray([label], np.long)
        # resize
        image = image.resize(self.resize_size)
        image = np.asarray(image, np.float32)

        size = image.shape
        image = image.transpose((2, 0, 1))
        x1 = random.randint(0, image.shape[1] - self.h)
        y1 = random.randint(0, image.shape[2] - self.w)
        image = image[:, x1:x1+self.h, y1:y1+self.w]

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)

        return image.copy(), label#.copy()


if __name__ == '__main__':
    dst = QDDataSet('../Warehouse/dataset/train/')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, label = data
        if i == 0:
            print(int(label))
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(np.uint8(img) )
            img.save('Demo.jpg')
        break
