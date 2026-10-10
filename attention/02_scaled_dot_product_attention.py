import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)

x_2 = inputs[1]
# 입력 임베딩 크기(3)
d_in = inputs.shape[1]
# 출력 임베딩 크기(2)
d_out = 2
torch.manual_seed(123)
# e.g.
# input                 W_query 
# [0.55, 0.87, 0.66] @ [ [0.2961, 0.5166],
#                        [0.2517, 0.6886],
#                        [0.0740, 0.8665] ]
#  첫 번째 원소:
# (0.55 * 0.2961) + (0.87 * 0.2517) + (0.66 * 0.0740)
# = 0.162855 + 0.218979 + 0.04884
# = 0.430674
# 두 번째 원소:
# (0.55 * 0.5166) + (0.87 * 0.6886) + (0.66 * 0.8665)
# = 0.28413 + 0.599082 + 0.57189
# = 1.455102
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print(keys)
print("values.shape: ", values.shape)
print(values)

# attention w22
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# attention
attn_score_2 = query_2 @ keys.T
print(attn_score_2)

# pic 3-16
              # python idiom last idx
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# context vector
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)