import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2D 전치(.T)에서 3D 배치 행렬곱을 위한 transpose(1, 2)로 변경
        # (2, 3, 4) 크기의 keys 텐서에서 1번과 2번 차원을 교체하여 (2, 4, 3)으로 변환 후 queries와 행렬곱 수행
        attn_scores = queries @ keys.transpose(1,2)
        # 마스크 적용: 미래 토큰 위치의 점수를 -무한대(-inf)로 채움
        # Softmax를 거쳐 -inf는 0으로 만든다
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # 어텐션 가중치에 Dropout 적용
        # 계산된 확률 분포(attn_weights)에 드롭아웃을 적용해 특정 단어에만 과도하게 집중되는 현상 방지
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec