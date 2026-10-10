from self_attention_v1 import SelfAttention_v1
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)

# 입력 임베딩 크기(3)
d_in = inputs.shape[1]
# 출력 임베딩 크기(2)
d_out = 2

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# 함수의 포인터 같은 객체로 생성 가능(__call__)
print(sa_v1(inputs))