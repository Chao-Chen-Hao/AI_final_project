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
from utils import generate_noise

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (1) x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 14 x 14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128) x 7 x 7
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            # state size. (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

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

class main():

    def __init__(self, batch_size=64, z_dim=100, epochs=100, lr=0.001, img_size=64):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # gpu device
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.epochs = epochs
        self.lr = lr
        self.G = Generator(self.z_dim).to(self.device)
        self.D = Discriminator().to(self.device)
        self.criterion = nn.BCELoss()
        self.criterion_MSE = nn.MSELoss()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr = self.lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr = self.lr)
        self.G_losses = []
        self.D_losses = []
        self.img_list = []
        self.img_size = img_size
        self.interp = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)

    def get_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset = QDDataSet(root='../Warehouse/dataset/train/', t_set='train', max_iters=self.epochs * 3000,
                resize_size=(self.img_size, self.img_size), crop_size=(self.img_size, self.img_size), mirror=False),
            batch_size = self.batch_size,
            shuffle=True
        )

        return train_loader

    def D_train(self, x):

        self.D.zero_grad()
        x_real = x.to(self.device)
        y_real = torch.ones(self.batch_size, 5, 5).to(self.device)
        x_real_predict = self.D(self.interp(x_real))
        D_real_loss = self.criterion(x_real_predict.view(-1), y_real.view(-1))
        D_real_loss.backward()

        noise = generate_noise(0, 0, noise_size=500, seed=123, gen=1, batch=self.batch_size).to(self.device)
        y_fake = torch.zeros(self.batch_size, 5, 5).to(self.device)
        x_fake = self.G(noise)
        x_fake_predict = self.D(self.interp(x_fake))
        D_fake_loss = self.criterion(x_fake_predict.view(-1), y_fake.view(-1))
        D_fake_loss.backward()

        D_total_loss = D_real_loss + D_fake_loss
        self.D_optimizer.step()

        print("D_real_loss: ", D_real_loss.item())
        print("D_fake_loss: ", D_fake_loss.item())

        return D_total_loss.data.item()      

    def G_train(self, x):
        self.G.zero_grad()
        noise = generate_noise(0, 0, noise_size=500, seed=123, gen=1, batch=self.batch_size).to(self.device)
        y_target = torch.ones(self.batch_size,5,5).to(self.device)
        x_fake = self.G(noise)
        y_fake = self.D(self.interp(x_fake))
        G_loss = self.criterion(y_fake.view(-1), y_target.view(-1))

        loss = G_loss
        loss.backward()
        self.G_optimizer.step()

        return G_loss.data.item()

    def Draw_plot(self):
        
        plt.figure()
        plt.title(" D Loss, G Loss / Iteration ")
        plt.plot(self.G_losses, label='G')
        plt.plot(self.D_losses, label='D')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('loss.png')

    def Draw_Anim_Image(self):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000, blit=True)
        ani.save('test.gif',writer='imagemagick', dpi=100 )
        plt.savefig('ani.gif')

    def Train(self):
        print("\nBasic GAN Implement\n")
        print("Star Training\t",end='')
        if self.device.type == 'cuda':
            print("\tUse {}\n".format(torch.cuda.get_device_name(0)))
        else:
            print("\tUse CPU\n")
        
        train_loader = self.get_dataloader()
        
        for epoch in range(self.epochs):    
            print("Epoch: ", epoch)

            for id, (train_x, train_y) in enumerate(train_loader):
                if(len(train_x) != self.batch_size):
                    continue
                print('[{:03}/{:03}]\t[{:03}/{:03}]\t Loss D: {:.4f} \tLoss G: {:.4f}'.format(epoch+1, self.epochs, id, len(train_loader), np.mean(self.D_losses), np.mean(self.G_losses)))
                if (epoch % 40 < 20):
                    self.D_losses.append( self.D_train(train_x.cuda()) )
                else:
                    self.D_losses.append( self.D_train(train_x.cuda()) )
                    self.G_losses.append( self.G_train(train_x.cuda()) )
            
            if (epoch % 20 == 0):
                torch.save(self.D.state_dict(), 'snapshots/model_D.pth')
                torch.save(self.G.state_dict(), 'snapshots/model_G.pth')
                with( torch.no_grad() ):
                    noise = torch.tensor(torch.randn(self.batch_size, self.z_dim, 1 , 1, device=self.device))
                    fake = self.G(noise).detach().cpu()
                    for i in range(5):
                        save_image(self.interp(fake)[i], 'img/'+str(epoch)+'_'+str(i)+'.png')
                                
            self.Draw_plot()

if __name__ == '__main__':
    train = main(batch_size=1000, z_dim=500, epochs=1000, lr=0.001, img_size=64).Train()