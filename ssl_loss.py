import torch
import torch.nn as nn
import torch.nn.functional as F

# Normalized Temperature-scaled Cross Entropy Loss
class NT_Xent(nn.Module):
    def __init__(self, batch_size=128, temperature=1):
        super(NT_Xent, self).__init__()
        self.batch_size  = batch_size # ミニバッチサイズ
        self.temperature = temperature # 温度パラメータ

        self.mask = self.mask_correlated_samples(self.batch_size) # マスクの作成
        self.criterion    = nn.CrossEntropyLoss(reduction="sum") # クロスエントロピー(softmaxを内包)
        self.similarity_f = nn.CosineSimilarity(dim=2) # コサイン類似度

    # 同一特徴量間の類似度とポジティブペアの類似度を削除するためのマスクの作成
    def mask_correlated_samples(self, batch_size):
        
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0) # 同一特徴量間の類似度が入っている位置の値を0に
        for i in range(batch_size):
            mask[i, batch_size+i] = 0 # ポジティブペアの類似度が入っている位置の値を0に
            mask[batch_size+i, i] = 0 # ポジティブペアの類似度が入っている位置の値を0に
        
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0) # ネットワークの２つの出力を1つのTensorに
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature # 全ての特徴量間の類似度を計算

        sim_i_j = torch.diag(sim,  self.batch_size)  # ポジティブペアの類似度(i->j))を抽出
        sim_j_i = torch.diag(sim, -self.batch_size)  # ポジティブペアの類似度(j->i))を抽出
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # ポジティブペアの類似度を1つのTensorに
        negative_samples = sim[self.mask].reshape(N, -1)  # ネガティブペアの類似度のみのTensorを作成

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # ポジティブペアとネガティブペアの類似度を1つのテンソルに
        labels = torch.zeros(N).to(positive_samples.cuda()).long()  # ポジティブペアの類似度の位置を表すTensorを作成

        loss = self.criterion(logits, labels)  # 総和を計算
        loss /= N  # 最終的な損失
        
        return loss


# Negative Cosine Similarity Loss
class Negative_CosSim(nn.Module):
    def __init__(self, dim=1):
        """
        Args:
            dim (float): 次元数
        """
        super(Negative_CosSim, self).__init__()
        self.dim = dim
        self.cos_sim = nn.CosineSimilarity(dim=self.dim).to('cuda')

    def forward(self, p1, p2, z1, z2):

        loss1 = self.cos_sim(p1, z2).mean() # View1のPredictor出力とView2のProjector出力の類似度
        loss2 = self.cos_sim(p2, z1).mean() # View2のPredictor出力とView1のProjector出力の類似度
        
        loss = -(loss1 + loss2) * 0.5 # 最終的な損失

        return loss