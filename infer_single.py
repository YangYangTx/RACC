'''
@File    :   train.py    
@Contact :   yangyang6@corp.netease.com

@Modify Time      @Author    @Version   @Desciption
------------      ---------  --------   -----------
2023/3/5 19:08   yangyang6     1.0        None
'''
# -*- coding: utf-8 -*-
import os
import cv2
import csv
import sys
import glob
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets
import functools
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import default_collate
from model import bqvsModel

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4,5,6,7'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'




parser = argparse.ArgumentParser(description='PyTorch  PBVS  Classification')

parser.add_argument('--num_classes', default=10, help='choose model type')
parser.add_argument('--checkpoint_path', type=str, default="checkpoint-EO-v2/task2_swa_model.pth")
parser.add_argument('--testPath', default="/mnt/dl-storage/dg-cephfs-0/public/Ambilight/PBVS/track2_test_march_1/EO_test", help='test image path')
parser.add_argument('--resultPath', default="submitfinal/0302_results_task2_best_senet_EO_v2.csv", help='test image path')

args = parser.parse_args()

class submitResult:
    def __init__(self): 
        
        #=========================== image init======================
        self.checkpoint_path = args.checkpoint_path
        self.testPath = args.testPath
        self.tempfile = args.resultPath
     
     
        clip_mean = [0.485, 0.456, 0.406]
        clip_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=clip_mean,
                                         std=clip_std)
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        
        self.model = bqvsModel("resnet50", 10, is_pretrained=True)
        self.model = torch.nn.DataParallel(self.model).cuda()
        
        if "swa" in self.checkpoint_path:
            # self.model.load_state_dict({"module." + k:v for k,v in torch.load('checkpoint-EO-v2/task2_swa_model.pth').items()})
            self.model.load_state_dict({"module." + k:v for k,v in torch.load(self.checkpoint_path).items()})
        elif "best" in self.checkpoint_path:
            # self.model.load_state_dict({k:v for k,v in torch.load('checkpoints-v2/model_best.pth.tar')["state_dict"].items()})
            self.model.load_state_dict({k:v for k,v in torch.load(self.checkpoint_path)["state_dict"].items()})
        else:
            print ("load model filed, please check result!")
        self.model.eval()


    def Infer_online(self):
        # file = open('submitfinal/0302_results_task2_best_senet_EO_v2.csv', 'w', newline='')
        self.tempfile_ = "".join(self.tempfile.split("/")[:-1])
        if not os.path.exists(self.tempfile_):
            os.mkdirs(self.tempfile_)

        file = open(self.tempfile, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['image_id', 'class_id', 'score'])
        #outputs = []
        imgpaths = glob.glob(self.testPath + "/*")
        imgpaths.sort()
        with torch.no_grad():
            for imgpath in imgpaths:
                img = self.val_transform(Image.open(imgpath).convert('RGB')).unsqueeze(0)
                img = img.float().cuda()
                output = self.model(img)
                soft = F.softmax(output, dim=1)
                score, predicted = torch.max(soft.data, 1)
                name = imgpath.split("/")[-1]
                #  check type
                writer.writerow([int(name[6:-4]), int(predicted.detach().cpu().numpy().tolist()[0]), float(score.detach().cpu().numpy().tolist()[0])])
        #torch.save(torch.cat(outputs, dim = 0).cpu(), 'val_embedding.tensor')
        

if __name__ == '__main__':
    imgLowqRecogstr = submitResult()
    st = time.time()
    imgLowqRecogstr.Infer_online()
    print ("spend all time is:", time.time() - st)
