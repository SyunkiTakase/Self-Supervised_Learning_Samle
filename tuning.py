import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from base_model import FeatureExtractor
from trainer import train, validation
from ssl_loss import NT_Xent, Negative_CosSim

def setting_model(device, class_names, method, tuning, weight):
    
    if method == 'SimCLR':
        model = FeatureExtractor(method='SimCLR', tuning=tuning, num_classes=len(class_names)).to(device) # エンコーダ
        checkpoint = torch.load(weight, map_location='cpu')
        checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['projector.weight', 'projector.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f'Removing key {k} from pretrained checkpoint')
                del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)
    
    elif method == 'SimSiam':
        model = FeatureExtractor(method='SimSiam', tuning=tuning, num_classes=len(class_names)).to(device) # エンコーダ
        checkpoint = torch.load(weight, map_location='cpu')
        checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['projector.weight', 'projector.bias', 'predictor.weight', 'predictor.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f'Removing key {k} from pretrained checkpoint')
                del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)

    return model

def tuning_layer(model, tuning):
    
    if tuning == 'all':
        params = model.parameters()
    
    elif tuning == 'classifier':
        # model全体を凍結する
        for param in model.parameters():
            param.requires_grad = False
        # 分類層(classifier)のみ学習可能にする
        for param in model.classifier.parameters():
            param.requires_grad = True
        params = model.classifier.parameters()
    
    return params
    
def setting_augment(img_size, use_deg):

    c_w = 0.2 # color_jitterの強さの調整
    color_jitter = transforms.ColorJitter(0.2*c_w, 0.2*c_w, 0.2*c_w, 0.2*c_w) # ランダムに明るさ,コントラスト,彩度,色相を変化
    train_transforms = transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.3), # color_jitter
        transforms.RandomGrayscale(p=0.1), # グレースケール
        transforms.RandomHorizontalFlip(), # 左右反転
        transforms.ToTensor(), # 画像をtensor化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正規化
        ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(), # 画像をtensor化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正規化
        ])

    return train_transforms, val_transforms
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('model_deg', exist_ok=True)

    # ハイパーパラメータ
    num_epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    img_size = args.img_size
    dataset_name = args.dataset
    method = args.method
    use_deg = args.use_deg
    tuning = args.tuning
    weight = args.weight 

    train_transforms, val_transforms = setting_augment(img_size, use_deg)
    
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transforms) 
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,  download=True, transform=val_transforms) 
    
    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,  download=True, transform=train_transforms) 
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,  download=True, transform=val_transforms) 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True) # データローダー
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False) # データローダー
    class_names = train_dataset.classes

    model = setting_model(device, class_names, method, tuning, weight) # モデルの設定
    params = tuning_layer(model, tuning) # 学習するパラメータ
    optimizer = torch.optim.Adam(params, lr=lr) # Optimizer
    criterion = torch.nn.CrossEntropyLoss() # 損失関数
    print('Model:', model)
    
    # 学習対象のパラメータを可視化
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {param_count}')
    
    # 学習ループ
    for epoch in range(num_epoch):
        train_loss, train_count = train(device, train_loader, model, criterion, optimizer, epoch)
        val_loss, val_count = validation(device, val_loader, model, criterion)

        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f} ')
        print(f'Epoch [{epoch+1}/{num_epoch}], Trainig Acc: {train_count/len(train_loader.dataset):.4f}, Validation Acc: {val_count/len(val_loader.dataset):.4f} ')

        if (epoch+1) % 10 == 0:
            print('saved!!')
            save_model_path = './model/' + str(method) + '_' + str(tuning) + '_' + str(epoch + 1) + '.tar'
            # save_model_path = os.path.join('model/','{}.tar'.format(epoch + 1))
            torch.save({
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch
            },save_model_path)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='Epoch数')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-3, help='学習率')
    parser.add_argument('--img_size', type=int, default=32, help='画像サイズ')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10', help='使用するデータセット')
    parser.add_argument('--method', type=str, choices=['SimCLR', 'SimSiam'], default='SimCLR', help='事前学習に使用した自己教師あり学習の手法')
    parser.add_argument('--tuning', type=str, choices=['all', 'classifier'], default='all', help='ファインチューニングか転移学習化を選択')
    parser.add_argument('--weight', type=str, help='事前学習モデルのパス')
    args=parser.parse_args()
    main(args)

