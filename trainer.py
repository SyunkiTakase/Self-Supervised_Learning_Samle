import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm

from functools import partial

def train_simclr(device, train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    sum_loss = 0.0

    for images, _ in tqdm(train_loader):
        view1, view2 = images # データ拡張済みの画像 -> shape:[batch, C, H, W]

        view1, view2 = view1.to(device), view2.to(device)
        view = torch.cat([view1, view2], dim=0) # [2 * batch, C, H, W]

        # 特徴量抽出
        features = model.forward_simclr(view)

        # NT-Xentの計算
        loss = criterion(features)

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