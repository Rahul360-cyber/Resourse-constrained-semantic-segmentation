
import os
import random

import torch
from torch import optim, squeeze
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info
import tqdm
import numpy as np

from model import BiSeNet
from model import ENet
from model import ICNet
from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb
from utils import scores
exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()

def main():

    cfg_file = open('codes/config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []
    if cfg.TRAIN.STAGE=='all':
        net=ENet(only_encode=False)
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE =='encoder':
        net=ENet(only_encode=True)
    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    '''criterion = torch.nn.BCEWithLogitsLoss().cuda() '''# Binary Classification
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)
    
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].tic()
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))


def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        '''labels = Variable(labels).cuda().float()''' # binary
        labels = Variable(labels.long()).cuda()
   
        optimizer.zero_grad()
        outputs=net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()



def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    total = 0
    correct = 0
    num_classes = 5
    class_total = [0] * num_classes
    class_correct = [0] * num_classes
    iou_per_class = [0.0] * num_classes
    label_trues = []
    label_preds = []

    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        label_trues.append(labels.cpu().numpy())
        label_preds.append(predicted.cpu().numpy())
        
        for c in range(num_classes):
            class_total[c] += torch.sum(labels == c).item()
            class_correct[c] += torch.sum((predicted == labels) & (predicted == c)).item()
    
    label_trues = np.concatenate(label_trues)
    label_preds = np.concatenate(label_preds)
    
    accuracy_scores, class_iou  = scores(label_trues, label_preds, num_classes)
    print(scores(label_trues, label_preds, num_classes))
    print('*********************')
    net.train()
    criterion.cuda()

if __name__ == '__main__':
  main()
  







