import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)

# 두 번째 input을 쿼리 토큰으로 사용
# 이 단어와 주변 단어들과의 관계를 묻는 것. 내적을 통해 관계도를 파악한다
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
print("raw: ", attn_scores_2)
for i, x_i in enumerate(inputs):
    # 벡터의 내적 계산
    attn_scores_2[i] = torch.dot(x_i, query)
print("내적 계산 후", attn_scores_2)

# query와 input을 내적 계산 후 정규화(scalar 값으로 변환)
normalize_attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("어텐션 가중치: ", normalize_attn_weights_2)
print("합: ", normalize_attn_weights_2.sum())

ctx_query = inputs[1]
ctx_vec_2 = torch.zeros(ctx_query.shape)
for i, input_i in enumerate(inputs):
    # attn_weights_2[i]: scalar 값이며, query와 얼마나 관계있는지를 나타내는 비율(.sum() = 1)
    # input_i: input을 임베딩한 정보(n차원 벡터)
    # scalar와 vector를 곱하면 vector의 방향은 유지한 채, 크기만큼 비율이 scaling
    ctx_vec_2 += normalize_attn_weights_2[i] * input_i
print(ctx_vec_2)

print("=========================================")

attn_scores = torch.empty(6, 6)
# for i, input_i in enumerate(inputs):
#     for j, input_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(input_i, input_j)
# 행렬 곱셈을 하는 동일 동작 (PyTorch Tensor matmul 사용)
# 1번, 2번, 3번... 6번 토큰 전체를 동시에 각각 쿼리로 올려놓고, 전체 어텐션 점수판을 한 번에 만드는 연산
# e.g. 0행 (attn_scores[0]): 1번째 토큰(inputs[0])을 쿼리로 둔 내적 결과
# inputs.T는 행렬의 전치를 트리거한다
# inputs (6 x 3)
# ┌────────────────────────────────────┐
# │  -- Your    (0.43, 0.15, 0.89) --  │
# │  -- journey (0.55, 0.87, 0.66) --  │
# │  -- starts  (0.57, 0.85, 0.64) --  │
# │  -- with    (0.22, 0.58, 0.33) --  │
# │  -- one     (0.77, 0.25, 0.10) --  │
# │  -- step    (0.05, 0.80, 0.55) --  │
# └────────────────────────────────────┘
#              @
# inputs.T (3 x 6)
# ┌────────────────────────────────────────────────────────┐
# │   |        |        |        |        |        |       │
# │ Your    journey  starts    with     one      step      │
# │   |        |        |        |        |        |       │
# └────────────────────────────────────────────────────────┘
#              =
# attn_scores (6 x 6)
# ┌────────────────────────────────────────────────────────────────────┐
# │  Your•Your   Your•jour   Your•star  Your•with Your•one  Your•step  │
# │  jour•Your   jour•jour   jour•star  jour•with jour•one  jour•step  │
# │  star•Your   star•jour   star•star  star•with star•one  star•step  │
# │  with•Your   with•jour   with•star  with•with with•one  with•step  │
# │  one•Your    one•jour    one•star   one•with  one•one   one•step   │
# │  step•Your   step•jour   step•star  step•with step•one  step•step  │
# └────────────────────────────────────────────────────────────────────┘
#              =
# attn_scores (실제 결과 값)
# ┌────────────────────────────────────────────────────────────┐
# │   0.9995    0.9544    0.9422    0.4753    0.4576    0.6310 │
# │   0.9544    1.4950    1.4754    0.8434    0.7070    1.0865 │
# │   0.9422    1.4754    1.4570    0.8296    0.7154    1.0605 │
# │   0.4753    0.8434    0.8296    0.4937    0.3474    0.6565 │
# │   0.4576    0.7070    0.7154    0.3474    0.6654    0.2935 │
# │   0.6310    1.0865    1.0605    0.6565    0.2935    0.9450 │
# └────────────────────────────────────────────────────────────┘
attn_scores = inputs @ inputs.T
print(attn_scores)

# 정규화
attn_weights = torch.softmax(attn_scores, dim = -1)
print(attn_weights)

# 어텐션 가중치와 입력의 행렬 곱
all_ctx_vecs = attn_weights @ inputs
print(all_ctx_vecs)