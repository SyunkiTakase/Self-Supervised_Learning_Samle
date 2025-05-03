import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from functools import partial

def train(device, train_loader, model, criterion, optimizer, epoch):
    model.train()
    sum_loss = 0.0
    count = 0

    for img,label in tqdm(train_loader):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()
        
        logit = model(img)
        loss = criterion(logit, label)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    return sum_loss, count

def validation(device, test_loader, model, criterion):
    model.eval()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            logit = model(img)
            loss = criterion(logit, label)
            
            sum_loss += loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count

def train_simclr(device, train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    sum_loss = 0.0

    for images, _ in tqdm(train_loader):
        view1, view2 = images # データ拡張済みの画像 -> shape:[batch, C, H, W]

        view1, view2 = view1.to(device), view2.to(device)

        # 特徴量抽出
        z1, z2 = model.forward_simclr(view1, view2)

        # NT-Xentの計算
        loss = criterion(z1, z2)

        # 損失のバックプロパゲーションと最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        
    return sum_loss

def train_simsiam(device, train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    sum_loss = 0.0

    for images, _ in tqdm(train_loader):
        view1, view2 = images # データ拡張済みの画像 -> shape:[batch, C, H, W]
        view1, view2 = view1.to(device), view2.to(device)

        # 特徴量抽出
        p1, p2, z1, z2 = model.forward_simsiam(view1, view2)

        # Negative Cosine Similarity Lossの計算
        loss = criterion(p1, p2, z1, z2)

        # 損失のバックプロパゲーションと最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        
    return sum_loss