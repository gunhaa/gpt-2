import torch
import torch.nn as nn

class LayerNorm(nn.Module):
                      # 임베딩 차원
    def __init__(self, emb_dim):
        super().__init__()
        # 작은 상수(엡실론), 분모 0 방지
        self.eps = 1e-5
        # 입력과 차원이 동일한 훈련 가능한 파라미터
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased= False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift