import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from torch import optim
from torch.optim import Adam

import numpy as np
from unet_models import TernausNet
# from FusionNet import FusionGenerator

import xlsxwriter

from pathlib import Path

import cv2
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision.utils as v_utils

import os
import time
import glob

# from datetime import datetime

# import visdom

from typing import Dict, Tuple



# cuda_is_available = torch.cuda.is_available()

img_transform = Compose([
    # transforms.CenterCrop(500),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(),
    ToTensor(),
    Normalize(mean=[0.441849141063118,	0.292323069796036,	0.210383472302730], std=[0.300617504748123,	0.221409146058924,	0.165845252163602]),
])


class LossBCE:
    def __init__(self, jaccard_weight=1):
        self.nll_loss = nn.BCELoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = outputs
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))

        return loss

class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            # nll_weight = cuda(
            #     torch.from_numpy(class_weights.astype(np.float32)))
            nll_weight = torch.from_numpy(class_weights.astype(np.float32))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss

class Giana2017Dataset(Dataset):
    def __init__(self, img_root, gt_root):
        
        self.img_root = img_root
        self.gt_root = gt_root
       
        fileImgTotalName=os.path.join(self.img_root, "*.bmp")
        fileGtTotalName=os.path.join(self.gt_root, "*.bmp")

        self.img_filenames = glob.glob(fileImgTotalName)
        self.gt_filenames = glob.glob(fileGtTotalName)

        print('file count:', self.img_filenames.__len__())

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
       
        fileNameTmp = self.img_filenames[index]
        fileName = fileNameTmp[len(self.img_root):]

        # print(self.img_filenames[index])
        # print(self.gt_filenames[index])
        img = cv2.imread(self.img_filenames[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        imgShape=img.shape

        if imgShape[0] == 500:
            img = img[:, 37:537, :]
        else:
            img = img[37:537, :, :]
        img = img_transform(img)
  
        #####################################################
        target = cv2.imread(self.gt_filenames[index])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = target.astype(np.uint8)

        imgShape=target.shape

        if imgShape[0] == 500:
            target = target[:, 37:537, :]
        else:
            target = target[37:537, :, :]


        target_polly = target[:,:, 0] > 250
        target_polly = target_polly * 1
        target_normal = target[:,:, 0] > 33
        target_normal = target_normal * 1

        target_background = target[:, :, 0] < 5
        target_background = target_background * 1


        target_polly = target_polly
        target_normal = target_normal - target_polly


        target_class = target_polly
        target_class = target_class + target_normal * 2
        # target_class = target_class + target_background * 3
        # print(target_class.max())
        # target_class = np.expand_dims(target_class, 0)
        target_class = target_class.astype(np.float32)
        target_class=torch.from_numpy(target_class)
        target_class=target_class.type(torch.long)

        return img, target_class, fileName


def cyclic_lr(epoch, init_lr=1e-3, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    return lr

def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + epsilon

    return 2 * (intersection / union).mean()


def validation(model, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    losses = []

    dice = []

    imgCnt = 0
    for inputs, targets, fileName1 in valid_loader:
        # print(valid_loader.sampler.data_source.img_filenames[imgCnt])
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        outputs=F.log_softmax(outputs, dim=1)
        
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        outputs = outputs[:, 0, :, :]
        outputs = torch.squeeze(outputs)
        targets = torch.squeeze(targets)
        targets = targets == 1
        targets = targets.type(torch.float)

        dice += [get_dice(targets, outputs).item()]

        # full_fileName = './' + directory + '/' + str(fileName1[imgCnt])
        # # 이미지 저장
        # v_utils.save_image(outputs.cpu().data, full_fileName)

    valid_loss = np.mean(losses)  # type: float
    valid_dice = np.mean(dice)

    print('Valid loss: {:.5f}, dice: {:.5f}'.format(valid_loss, valid_dice))
    metrics = {'valid_loss': valid_loss, 'dice': valid_dice}
    return metrics

if __name__ == '__main__':

    print('excel file make')
    # make_dir("result/train")
    workbook = xlsxwriter.Workbook('validation_data.xlsx')
    worksheet = workbook.add_worksheet()
    
    def make_loader(train_root, target_root, shuffle = False):
        return DataLoader(
            dataset=Giana2017Dataset(train_root, target_root),
            shuffle=shuffle,
            num_workers=1,
            batch_size=4,
            pin_memory=True
        )
    def make_loader_Test(train_root, target_root, shuffle = False):
        return DataLoader(
            dataset=Giana2017Dataset(train_root, target_root),
            shuffle=shuffle,
            num_workers=1,
            batch_size=4,
            pin_memory=True
        )
    print("start")


    root_path = "GIANA2018_Aug_0912"
    train_root =        root_path + '/train'
    target_root =       root_path + '/train_labels'

    test_root =         root_path + '/test'
    testTarget_root =   root_path + '/test_labels'


    print("data load start")
    train_loader = make_loader(train_root, target_root, shuffle=True)
    valid_loader = make_loader_Test(test_root, testTarget_root)
    
    num_classes = 3
    model = TernausNet(num_classes)
    # model.cuda()
    # model = nn.DataParallel(model).cuda()
    model.train()

    # print("model load start")
    # weight_path = './weight/bestDice_TernausNet_0_55.pth'
    # model.load_state_dict(torch.load(weight_path))

    # train
    n_epochs = 1
    # n_epochs = 100
    
    epoch = 0
    step = 0

    # criterion = Loss()
    if num_classes > 1:
        criterion = LossMulti( jaccard_weight=0.2, class_weights=None, num_classes = num_classes)
        # criterion = nn.NLLLoss()
    else:
        criterion = LossBCE(0.2)
    
    log = 'train.log'
    
    # best_dice_loss = float('inf')
    # best_valid_loss = float(0)

    best_valid_loss = float('inf')
    best_dice = float(0)

    # lossFileName = open("/home/modulabs/heedongYoon/yoon_test/best_loss_save.txt", 'w')
    lossFileName = open("best_loss_save.txt", 'w')
    lr = 1e-4


    xlsRow = 0
    xlsCols = 0

    worksheet.write(xlsRow, 0, 'epoch')
    xlsRow = xlsRow + 1
    worksheet.write(xlsRow, 0, 'train loss')
    xlsRow = xlsRow + 1
    worksheet.write(xlsRow, 0, 'validation loss')
    xlsRow = xlsRow + 1
    worksheet.write(xlsRow, 0, 'validation dice')
    xlsRow = xlsRow + 1
    worksheet.write(xlsRow, 0, 'learning rate')

    for epoch in range(epoch, n_epochs + 1):

        # 그래프 그리기
        # vis = visdom.Visdom(port=10003)
        Yaxis = torch.ones(1)
        Xaxis = np.array([0])
        # plot = vis.line(Y=Yaxis, X=Xaxis)

        lr = cyclic_lr(epoch)
        # optimizer = Adam(model.parameters(), lr, weight_decay=0.0005)
        optimizer = Adam(model.parameters(), lr)

        print("epoch: ",epoch)
        print("lr: ",lr)
        
        losses = []
        tl = train_loader

        try:
            for i, (inputs, targets, fileName1) in enumerate(tl):
                # inputs = inputs.cuda()
                # targets = targets.cuda()
                
                outputs = model(inputs)
                if num_classes > 1:
                    outputs=F.log_softmax(outputs, dim=1)

                loss=criterion(outputs, targets)

                losses.append(loss.item())
                Yaxis = loss.data[0].unsqueeze(0).cpu()
                Xaxis = np.array([i])
                # vis.line(Y=Yaxis, X=Xaxis, win=plot, update='append',
                #             opts=dict(title=str(epoch),
                #             showlegend=True))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
          
                print('Loss: {}'.format(loss.item()))
                # write_event(log, step, loss=loss)

            valid_metrics = validation(model, criterion, valid_loader)
            valid_loss = valid_metrics['valid_loss']
            dice_value = valid_metrics['dice']


            if valid_loss < best_valid_loss:

                lossNp = loss.data.cpu().numpy()
                lossFloat=float(lossNp)
                lossStr = str(lossNp)

                valid_lossStr = str(valid_loss)

                dice_valueStr = str(dice_value)

                lrStr = str(lr)

                # text_loss = vis.text("Epoch(Best): " + str(epoch) + ",loss: " + lossStr + ",valid_loss: " + valid_lossStr + ",valid_dice: " + "/n" + dice_valueStr + ",lr: " + lrStr)
                best_valid_loss = valid_loss
                # torch.save(model.state_dict(), 'best_TernausNet.pth')
                torch.save(model.state_dict(), 'bestValid_TernausNet.pth')

                epochStr = str(epoch)
                # data = "epoch: " + epochStr + ", loss: " + lossStr + "\n"
                data = "epoch: " + epochStr  + "\n"
                lossFileName.write(data)
                data = "loss: " + lossStr  + "\n"
                lossFileName.write(data)
                data = "valid_loss: " + valid_lossStr  + "\n"
                lossFileName.write(data)
                data = "valid_dice: " + dice_valueStr  + "\n"
                lossFileName.write(data)
                data = "lr: " + lrStr  + "\n"
                lossFileName.write(data)
                lossFileName.write("------------"  + "\n")
            else:

                lossNp = loss.data.cpu().numpy()
                lossFloat = float(lossNp)
                lossStr = str(lossNp)

                valid_lossStr = str(valid_loss)

                dice_valueStr = str(dice_value)

                lrStr = str(lr)

                # text_loss = vis.text("Epoch(Best): " + str(epoch) + ",loss: " + lossStr + ",valid_loss: " + valid_lossStr + ",valid_dice: " + dice_valueStr + ",lr: " + lrStr)


            xlsCols = xlsCols + 1
            worksheet.write(0, xlsCols, epoch)
            lossTrainMean=np.mean(losses)
            worksheet.write(1, xlsCols, lossTrainMean)
            worksheet.write(2, xlsCols, valid_loss)
            worksheet.write(3, xlsCols, dice_value)
            worksheet.write(4, xlsCols, lr)

            if (dice_value > best_dice):

                # vis.text("best_dice", win=text_loss, append=True)

                best_dice = dice_value
                torch.save(model.state_dict(), 'bestDice_TernausNet.pth')


            if (epoch % 10) == 0:
                print((str(epoch)+'_TernausNet.pth'))
                torch.save( model.state_dict(), (str(epoch)+'_TernausNet.pth') )
            
        except KeyboardInterrupt:
            # tq.close()
            print('Ctrl+C, saving snapshot')
            print('done.')
            lossFileName.close()
            workbook.close()
            
    lossFileName.close()
    workbook.close()