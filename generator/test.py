import torch
from torch import nn
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
from dataset import QDDataSet
from utils import generate_noise, get_labels
import os

class_list = {0:'airplane', 1:'bee', 2:'bicycle', 3:'bird', 4:'butterfly', 5:'cake',
    6:'camera', 7:'cat', 8:'chair', 9:'clock', 10:'computer',
    11:'diamond', 12:'door', 13:'ear', 14:'guitar', 15:'hamburger',
    16:'hammer', 17:'hand', 18:'hat', 19:'ladder', 20:'leaf',
    21:'lion', 22:'pencil', 23:'rabbit', 24:'scissors', 25:'shoe',
    26:'star', 27:'sword', 28:'The Eiffel Tower', 29:'tree'}

class Generator(nn.Module):
    def __init__(self, z_dim):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (256) x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (128) x 8 x 8
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 16 x 16
            nn.ConvTranspose2d( 64, 1, 4, 2, 3, bias=False),
            # state size. (64) x 28 x 28
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
def evaluate(batch_size=300, z_dim=500, img_size=64, num_img=10, threshold=0.9, file_name='../Warehouse/dataset/demo.csv', model_path='snapshots/model_G.pth'):
    if not os.path.exists('output/image'):
        os.makedirs('output/image')
    if not os.path.exists('output/csv'):
        os.makedirs('output/csv')
    
    interp = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    G = Generator(z_dim).to(device)
    saved_state_dict = torch.load(model_path, map_location='cuda:0')
    G.load_state_dict(saved_state_dict)
    label = get_labels(file_name)
    for c in range(len(label)):
        for i in range(num_img):
            noise = generate_noise(label[c], i)
            fake = G(noise.to(device, dtype=torch.float)).detach().cpu()
            fake = interp(fake)[0]
            fake = (fake - torch.min(fake).item())
            fake = fake / torch.max(fake).item()
            fake[fake < threshold] = 0.0
            fake[fake >= threshold] = 1.0
            name = class_list[label[c]] + '_'+str(i+1)
            save_image(fake[0], 'output/image/' + name +'.png')
            fake[0].numpy().tofile('output/csv/' + name + '.csv')
            #file_csv = open('output/csv/' + name + '.csv',"w+")
            #file_csv.write(fake[0].numpy().tostring)#np.array2string(fake[0].numpy(), max_line_width=20000)
            #file_csv.close()
    
if __name__ == '__main__':
    evaluate(file_name='../Warehouse/dataset/demo.csv', model_path='snapshots/model_G.pth')