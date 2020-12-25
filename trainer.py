import torch.nn as nn
from torch.utils import data, model_zoo
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn.init as init
import copy
import numpy as np
from efficientnet_pytorch import EfficientNet

class QDTrainer(nn.Module):
    def __init__(self, args):
        super(QDTrainer, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=30)
        if args.restore_from != None:
            saved_state_dict = torch.load(args.restore_from)
            self.model.load_state_dict(saved_state_dict)
        self.model = self.model.cuda()
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters: " + str(pytorch_total_params))

        self.gen_opt = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
        self.loss = nn.CrossEntropyLoss()
        self.sm = torch.nn.Softmax(dim = 1)
        
    def gen_update(self, images, labels, i_iter):
        self.gen_opt.zero_grad()

        pred = self.sm(self.model(images))

        loss = self.loss(pred, labels)
        loss.backward()

        self.gen_opt.step()

        return loss
