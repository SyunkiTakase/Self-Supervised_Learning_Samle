import torch
import torch.nn as nn
import torch.nn.functional as F

# Normalized Temperature-scaled Cross Entropy Loss
class NT_Xent(nn.Module):
    def __init__(self, tmp=1.0):
        """
        Args:
            tmp (float): 温度パラメータ
        """
        super(NT_Xent, self).__init__()
        self.tmp = tmp

    def forward(self, embeddings):
        batch_size = embeddings.shape[0] // 2  # バッチサイズ

        # 内積による類似度行列を計算
        similarity_matrix = torch.mm(embeddings, embeddings.T) / self.tmp  # [2 * batch, 2 * batch]

        # 対角成分を除外するためのマスク
        mask = torch.eye(2 * batch_size, device=embeddings.device, dtype=torch.bool)
        
        # log-softmax の適用
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # 正のペアのインデックス
        positive_indices = torch.arange(batch_size, device=embeddings.device)
        positive_indices = torch.cat([positive_indices + batch_size, positive_indices])

        # 正例の対数確率を抽出
        loss = -log_prob[torch.arange(2 * batch_size, device=embeddings.device), positive_indices]

        return loss.mean()
