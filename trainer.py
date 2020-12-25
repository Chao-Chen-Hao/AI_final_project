import torch.nn as nn
from torch.utils import data, model_zoo
import torch.optim as optim
import torch.nn.functional as F
from modeling.deeplabv2 import Deeplabv2
import torch
import torch.nn.init as init
import copy
import numpy as np
from efficientnet_pytorch import EfficientNet


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class Trainer(nn.Module):
    def __init__(self, args):
        super(AD_Trainer, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        #saved_state_dict = torch.load(args.restore_from,map_location='cuda:0')
        #self.model.load_state_dict(saved_state_dict)
        
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters: " + str(pytorch_total_params))
        
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.learning_rate}]

        self.gen_opt = optim.SGD(train_params, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        self.loss = nn.CrossEntropyLoss(ignore_index=255)
        self.sm = torch.nn.Softmax(dim = 1)
        self.model = self.model.cuda()
        self.interp = nn.Upsample(size= args.crop_size, mode='bilinear', align_corners=True)

    def update_label(self, labels, prediction):
        criterion = nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255, reduction = 'none')
        loss = criterion(prediction, labels)
        print('original loss: %f'% self.seg_loss(prediction, labels) )
        loss_data = loss.data.cpu().numpy()
        mm = np.percentile(loss_data[:], self.only_hard_label)
        labels[loss < mm] = 255
        return labels

    def gen_update(self, images, labels, i_iter):
        self.gen_opt.zero_grad()

        pred = self.G(images)
        pred = self.interp(pred)

        if self.class_balance:            
            self.seg_loss = self.update_class_criterion(labels)

        if self.only_hard_label > 0: # class balance
            labels = self.update_label(labels.clone(), pred)
            loss_seg = self.seg_loss(pred, labels)
        else:
            loss_seg = self.seg_loss(pred, labels)

        loss = loss_seg

        if self.fp16:
            with amp.scale_loss(loss, self.gen_opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.gen_opt.step()

        return loss_seg
