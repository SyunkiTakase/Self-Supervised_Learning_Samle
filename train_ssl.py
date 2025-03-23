import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from base_model import FeatureExtractor
from ssl_loss import NT_Xent, Consine_loss
from trainer import train_simclr, train_simsiam
from data_aug.contrastive_learning_dataset import SimCLRTransform, ContrastiveLearningDataset

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ハイパーパラメータ
    num_epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    img_size = args.img_size
    dataset_name = args.dataset
    tmp = args.tmp
    dim = args.dim
    use_subset = args.subset
    method = args.method

    if use_subset == True:
        transform = transforms.Compose([transforms.ToTensor()]) # データセットのTensor化
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # 1/1000にサブセット化
        subset_size = len(train_dataset) // 1000  # 1/1000サイズを計算
        indices = torch.randperm(len(train_dataset))[:subset_size]  # ランダムなインデックスを選択
        train_subset = torch.utils.data.Subset(train_dataset, indices)  # サブセットを作成

        # SimCLRのデータ拡張をサブセットに適用
        simclr_transform = SimCLRTransform(size=img_size, n_views=2)
        train_subset.dataset.transform = simclr_transform  # データセットの変換を更新
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True) # データローダー

    else:
        dataset = ContrastiveLearningDataset('./data')

        train_dataset = dataset.get_dataset('cifar10', n_views=2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) # データローダー

    if method == 'SimCLR':
        model = FeatureExtractor(mode='SimCLR').to(device) # エンコーダ
        criterion = NT_Xent(tmp=tmp) # 損失関数
    elif method == 'SimSiam':
        model = FeatureExtractor(mode='SimSiam').to(device) # エンコーダ
        criterion = Consine_loss(dim=dim) # 損失関数

    optimizer = optim.Adam(model.parameters(), lr=lr) # Optimizer
    print('Encoder:', model)
    
    # 学習ループ
    for epoch in range(num_epoch):
        if method == 'SimCLR':
            sum_loss = train_simclr.train(device, train_loader, model, criterion, optimizer, epoch)
        elif method == 'SimSiam':
            sum_loss = train_simsiam.train(device, train_loader, model, criterion, optimizer, epoch)

        print(f"Epoch [{epoch+1}/10], Loss: {sum_loss/len(train_loader):.4f}")

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--tmp", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--subset", action='store_true')
    parser.add_argument("--method", type=str, choices=['SimCLR', 'SimSiam'], default="SimCLR")
    args=parser.parse_args()
    main(args)

