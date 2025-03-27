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
        
        similarity_matrix = similarity_matrix * mask
        
        # log-softmax の適用
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # 正のペアのインデックス
        positive_indices = torch.arange(batch_size, device=embeddings.device)
        positive_indices = torch.cat([positive_indices + batch_size, positive_indices])

        # 正例の対数確率を抽出
        loss = -log_prob[torch.arange(2 * batch_size, device=embeddings.device), positive_indices]

        return loss.mean()


# Negative Cosine Similarity Loss
class Negative_CosineSimilarity(nn.Module):
    def __init__(self, dim=1):
        """
        Args:
            dim (float): 次元数
        """
        super(Negative_CosineSimilarity, self).__init__()
        self.dim = dim
        self.cos_sim = nn.CosineSimilarity(dim=self.dim).to('cuda')

    def forward(self, p1, p2, z1, z2):

        # View1のPredictor出力とView2のProjector出力の類似度
        loss1 = self.cos_sim(p1, z2).mean()
        
        # View2のPredictor出力とView1のProjector出力の類似度
        loss2 = self.cos_sim(p2, z1).mean() 
        
        # 最終的な損失
        loss = -( loss1 + loss2) * 0.5

        return loss.mean()