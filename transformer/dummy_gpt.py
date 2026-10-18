import tiktoken
import torch
import torch.nn as nn
from dummy_gpt_model import DummyGPTModel
from gpt_config_124m import GPT_CONFIG_124M

# 50257 voca
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# gpt2 tokenizer의 voca로 매핑 후 tensor(vector)화 시킨다
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
# logits 라는 모델의 출력
logits = model(batch)
print("출력 크기: ", logits.shape)
print(logits)

print("===============================")

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print(out)
# 수십 개의 트랜스포머 블록을 통과하면서 데이터가 폭주하거나 소멸하지 않도록, 매 단계마다 일정한 규격(평균 0, 분산 1)으로 고르게 고정해 주는 "안정장치" 역할
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("평균: \n", mean)
print("분산:\n", var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
print("정규화된 층 출력:\n", out_norm)
print("평균:\n", mean)
print("분산:\n", var)
torch.set_printoptions(sci_mode=False)
print("평균:\n", mean)
print("분산:\n", var)