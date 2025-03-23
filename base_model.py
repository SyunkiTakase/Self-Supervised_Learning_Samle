import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

# ResNet-18の特徴量抽出モデル
class FeatureExtractor(nn.Module):
    def __init__(self, method='SimCLR', num_classes=None):
        super(FeatureExtractor, self).__init__()

        # ResNet-50のベースモデル
        self.base_model = resnet50(pretrained=False) 
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # 最後の分類層を除外
        self.flatten = nn.Flatten() # 出力をフラット化
        self.proj_dim = 128 # Projectorの出力次元
        self.prev_dim = 512 # Predictorの入力次元
        self.out_dim = 2048 # Predictorの出力次元

        if method == 'SimCLR':
            # Projectorの定義
            dim_mlp = self.base_model.fc.in_features
            self.projector = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, self.proj_dim)
            )
            
        elif method == 'SimSimam':
            # Projectorの定義
            dim_mlp = self.base_model.fc.in_features
            self.projector = nn.Sequential(
                nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                nn.BatchNorm1d(self.prev_dim),
                nn.ReLU(inplace=True), 
                nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                nn.BatchNorm1d(self.prev_dim),
                nn.ReLU(inplace=True), 
                self.encoder.fc,
                nn.BatchNorm1d(self.out_dim, affine=False) 
            )
            self.predictor = nn.Sequential(
                nn.Linear(self.out_dim, self.prev_dim, bias=False),
                nn.BatchNorm1d(self.prev_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.prev_dim, self.out_dim)
            )

        else:
            # Fine Tuning用の出力層を定義
            self.fc = nn.Linear(dim_mlp, num_classes)

    def forward(self, x):
        # Encoder部分
        x = self.features(x)
        x = self.flatten(x)

        if self.fc is not None:
            x = self.fc(x)

        return x

    def forward_simclr(self, x):
        # Encoder部分
        x = self.features(x)
        x = self.flatten(x)

        # Projector部分
        x = self.projector(x)
        x = F.normalize(x, dim=1)

        return x

    def forward_simsiam(self, x1, x2):
        # Encoder部分
        z1 = self.features(x1)
        z2 = self.features(x2)

        # Projector部分
        z1 = self.projector(z1)
        z2 = self.projector(z1)
        
        # Predictor部分
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()