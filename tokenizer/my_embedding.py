import torch
from gpt_dataset_v1 import GPTDatasetV1

input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))

print("===================================")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
max_length = 4
dataloader = GPTDatasetV1.create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("token ID: \n ", inputs)
print("\ninput size: \n", inputs.shape)

# 50257의 단어사전, 256차원
# 핵심은 vocab에 해당하는 id를 n차원 백터로 만드는 것이다
token_vocab_size = 50257
token_output_dim = 256
token_embedding_layer = torch.nn.Embedding(token_vocab_size, token_output_dim)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# vocab의 임베딩 정보는 위치에 대한 정보를 포함하지 않기에 위치 정보를 임베딩한다
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, 256)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
print(pos_embeddings)

# 위치 정보를 인식하기 위해, vocab의 정보를한 임베딩 값에 pos를 임베딩 한 값을 더한다
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)