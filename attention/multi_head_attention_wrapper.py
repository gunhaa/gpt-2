import torch
import torch.nn as nn
from causal_attention import CausalAttention

# 여러 개의 CausalAttention 헤드를 병렬로 실행하여 텍스트를 다양한 관점에서 동시에 분석
# 각 헤드가 독립적으로 계산한 특징 텐서들을 리스트 컴프리헨션으로 모은 뒤,
# torch.cat을 통해 마지막 차원으로 concatenate 후 최종 멀티 헤드 출력을 만든다
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 현재는 가중치가 같게 head가 실행되지만, 이후에는 학습에(backpropagation) 따라 head마다 attention 가중치가 변함
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)