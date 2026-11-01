# 용어집 (Glossary)

- 마지막 갱신: 2026-11-01
- 학습 단계 순서로 정리했다. "코드" 열은 이 저장소에서 해당 개념이 등장하는 위치다(⬜는 앞으로 구현할 부분).

## 0. 기본 (PyTorch / 딥러닝)

| 용어 | 뜻 | 코드 |
|---|---|---|
| tensor | 다차원 배열. 모든 데이터와 가중치가 tensor다 | 전체 |
| shape | tensor 각 축의 크기. 예: `(batch, seq_len, emb_dim)` | 전체 |
| batch | 한 번에 같이 처리하는 샘플 묶음 | `gpt_dataset_v1.py` |
| parameter / weight | 학습으로 값이 바뀌는 숫자. `nn.Parameter`, `nn.Linear`의 가중치 | `layer_norm.py` |
| hyperparameter | 사람이 정하는 설정값. 층 수, 차원, dropout 비율 등 | `gpt_config_124m.py` |
| `nn.Module` | PyTorch 모델·층의 기본 클래스. `forward()`에 계산을 정의 | 전체 |
| forward / backward | 순전파(입력 → 출력 계산) / 역전파(gradient 계산) | ⬜ backward |
| training / inference | 학습(W를 수정) / 추론(고정된 W로 계산만, 실제 서비스) | ⬜ |
| gradient | loss를 줄이려면 각 가중치를 어느 방향으로 바꿀지 나타내는 값 | ⬜ |

## 1. 토큰화 (Tokenization)

| 용어 | 뜻 | 코드 |
|---|---|---|
| token | 모델이 다루는 텍스트의 최소 단위(단어·부분 단어·문자) | `my_token.py` |
| tokenizer | 텍스트 ↔ token ID 변환기. `encode` / `decode` | `simple_tokenizer_v1.py` |
| vocabulary (vocab) | token과 ID의 대응표. GPT-2는 50,257개 | `vocab_size` |
| token ID | vocab 안에서 token의 번호(정수) | |
| BPE (Byte Pair Encoding) | 자주 붙어 나오는 글자 쌍을 합쳐 부분 단어를 만드는 방식. 모르는 단어도 쪼개서 표현 가능 | `tiktoken` |
| special token | 특수 목적 token. `<\|endoftext\|>`(문서 경계), `<\|unk\|>`(모르는 단어) | `simple_tokenzier_v2.py` |
| context length | 모델이 한 번에 볼 수 있는 최대 token 수. GPT-2는 1024 | `context_length` |
| sliding window | 긴 텍스트를 일정 길이로 잘라 여러 샘플을 만드는 방식 | `sliding_window.py` |
| stride | window를 몇 token씩 옮길지 | `gpt_dataset_v1.py` |
| input / target | target은 input을 한 칸 오른쪽으로 민 것. "다음 token 맞히기" 정답 | `gpt_dataset_v1.py` |

## 2. 임베딩 (Embedding)

| 용어 | 뜻 | 코드 |
|---|---|---|
| embedding | 이산적인 ID를 연속적인 실수 벡터로 바꾼 것 | `my_embedding.py` |
| dense vector | 대부분 값이 0이 아닌 벡터 (↔ one-hot 같은 sparse vector) | |
| token embedding | token ID → 벡터. `nn.Embedding(vocab_size, emb_dim)`, 사실상 lookup table | `tok_emb` |
| positional embedding | 위치(0, 1, 2…) → 벡터. 어텐션만으로는 순서를 모르기 때문에 필요 | `pos_emb` |
| emb_dim (d_model) | 임베딩 및 hidden state의 차원. GPT-2 small은 768 | `emb_dim` |

## 3. 어텐션 (Attention)

| 용어 | 뜻 | 코드 |
|---|---|---|
| attention | 각 token이 다른 token들을 얼마나 참고할지 계산해 정보를 섞는 연산 | `01_basic_attention.py` |
| query / key / value (Q, K, V) | Q: 내가 찾는 것, K: 각 token의 색인, V: 실제로 가져올 내용 ([Q&A](attention/01-query-key-value.md)) | `W_query` 등 |
| soft lookup | 일치하는 하나가 아니라 모든 value를 점수 비율로 섞어 가져오는 조회. attention의 본질 | |
| attention score | Q·K 내적 값. 정규화 전의 관련도 | `attn_scores` |
| attention weight | score에 softmax를 적용한 값. 합이 1인 비율 | `attn_weights` |
| softmax | 실수 벡터를 합이 1인 확률 분포로 바꾸는 함수 | |
| scaled dot-product | score를 $\sqrt{d_k}$로 나눈 것. 값이 너무 커져 softmax가 한쪽으로 쏠리는 것을 방지 | `02_scaled_...py` |
| context vector | attention weight로 V를 가중합한 결과. 문맥이 반영된 벡터 | `context_vec` |
| self-attention | Q, K, V가 모두 같은 입력에서 나오는 어텐션 | `self_attention_v2.py` |
| causal (masked) attention | 미래 token을 보지 못하게 가리는 어텐션. GPT의 핵심 | `causal_attention.py` |
| mask | 가릴 위치 표시. 대각선 위를 `-inf`로 채워 softmax 후 0이 되게 함. 학습 때 모든 위치를 동시에 예측하므로 뒤 token(정답)을 보지 못하게 하려는 것 | `torch.triu` |
| dropout | 학습 중 일부 값을 무작위로 0으로 만들어 과적합을 방지 | `nn.Dropout` |
| multi-head attention | 어텐션을 여러 head로 나눠 병렬로 계산 후 합침. head마다 다른 관계를 학습. 비용은 head 1개와 같음 | `multi_head_attetion.py` |
| head_dim | head 하나의 차원 = `emb_dim / n_heads` = 768 / 12 = 64 | `head_dim` |
| out_proj | head들을 합친 뒤 섞어 주는 마지막 Linear. head마다 64개인 value 결과를 768개로 되돌림(3Blue1Brown의 value_up / output matrix) ([Q&A](attention/01-query-key-value.md)) | `out_proj` |
| low-rank | 중간 차원을 좁힌 변환. 768 → 64 → 768처럼 지나가면 parameter가 적다 | `head_dim` |
| attention pattern | token 수 × token 수 크기의 attention weight 표. 문맥 길이의 제곱으로 커짐 | `attn_weights` |

## 4. 트랜스포머 블록 / GPT 구조

| 용어 | 뜻 | 코드 |
|---|---|---|
| transformer block (layer) | attention + FeedForward + LayerNorm + residual로 이루어진 반복 단위 | 🧪 `DummyTransformerBlock` |
| stacking | 구조는 같고 가중치가 다른 블록을 차례로 쌓는 것. 재귀·분할 정복과 다름 ([Q&A](general/02-stacking-vs-recursion.md)) | `trf_blocks` |
| LayerNorm | token 하나의 벡터를 평균 0, 분산 1로 정규화한 뒤 scale·shift. 학습 안정화 | `layer_norm.py` |
| Pre-LN | LayerNorm을 attention/FFN **앞**에 두는 방식(GPT-2) | ⬜ |
| FeedForward (FFN, MLP) | token별로 따로 적용하는 2층 신경망. 768 → 3072 → 768 | ⬜ |
| GELU | GPT가 쓰는 활성화 함수. ReLU보다 매끄러움 | ⬜ |
| residual (shortcut) connection | 블록 입력을 출력에 더하는 것(`x + f(x)`). 깊은 망에서도 gradient가 잘 흐름 | ⬜ |
| hidden state | 모델 중간에서 token마다 나오는 벡터. 블록마다 갱신됨 ([Q&A](general/01-semantic-embedding-vs-gpt.md)) | `x` in `forward` |
| final norm | 마지막 블록 뒤의 LayerNorm | `final_norm` |
| out_head (LM head) | 최종 hidden state → vocab 크기 logits로 바꾸는 Linear | `out_head` |
| logits | softmax 전의 점수. token마다 vocab 50,257개 각각의 점수 | `logits` |
| 124M | GPT-2 small의 파라미터 수(약 1억 2400만 개) | `GPT_CONFIG_124M` |

## 5. 모델 구조 분류

| 용어 | 뜻 |
|---|---|
| encoder | 양방향 어텐션으로 입력 전체를 이해하는 부분 |
| decoder | causal 어텐션으로 한 token씩 생성하는 부분 |
| encoder-only | BERT, KURE-v1. 분류·임베딩에 사용 |
| decoder-only | GPT 계열. 생성에 사용 (← 지금 만드는 것) |
| encoder-decoder | 원래 Transformer, T5. 번역·요약 |
| autoregressive | 자기가 만든 출력을 다음 입력으로 넣어 한 token씩 생성하는 방식 |

## 6. 학습 (Training) — ⬜ 앞으로

| 용어 | 뜻 |
|---|---|
| pretraining | 대량 텍스트로 다음 token 예측을 학습하는 1단계 |
| loss | 정답과 얼마나 다른지 나타내는 값. 줄이는 것이 목표 |
| cross-entropy | 분류용 loss. 정답 token의 확률이 낮을수록 커짐 |
| perplexity | `exp(loss)`. "평균적으로 몇 개 후보 중에서 헷갈리는가"로 해석 |
| optimizer (AdamW) | gradient로 가중치를 갱신하는 알고리즘 |
| learning rate | 한 번에 가중치를 얼마나 크게 바꿀지 |
| epoch | 전체 데이터를 한 바퀴 학습한 단위 |
| overfitting | 학습 데이터만 외워 새 데이터에서 성능이 떨어지는 현상 |
| weight tying | token embedding과 out_head가 같은 가중치를 공유하는 기법(GPT-2 원본) |
| fine-tuning | 사전 훈련된 모델을 특정 작업에 맞게 추가 학습 |
| instruction tuning | 지시-응답 데이터로 fine-tuning해서 지시를 따르게 만드는 것 |

## 7. 생성 (Generation) — ⬜ 앞으로

| 용어 | 뜻 |
|---|---|
| decoding | logits에서 다음 token을 고르는 방법 |
| greedy decoding | 항상 확률이 가장 높은 token 선택 |
| temperature | logits를 나누는 값. 낮으면 보수적, 높으면 다양한 출력 |
| top-k sampling | 상위 k개 후보 중에서만 확률적으로 선택 |
| KV cache | 생성 중 이전 token들의 key·value를 저장해 재사용. 새 token의 q·k·v만 계산 ([Q&A](attention/01-query-key-value.md)) |

## 8. 임베딩 모델 / 검색 (참고)

| 용어 | 뜻 |
|---|---|
| semantic (sentence) embedding | 문장 의미를 담은 벡터 1개. 검색·유사도에 사용 |
| pooling | token별 hidden state를 벡터 1개로 줄이는 방법: CLS / mean / last-token |
| contrastive learning | 비슷한 쌍은 가깝게, 다른 쌍은 멀게 학습 |
| hard negative | 정답과 헷갈릴 만큼 비슷하지만 오답인 예시 |
| cosine similarity | 두 벡터의 방향이 얼마나 비슷한지(-1 ~ 1) |
| RAG | 검색한 문서를 LLM 입력에 넣어 답하게 하는 방식 |
| multi-vector (ColBERT) | 문장당 벡터 1개가 아니라 token마다 벡터를 남겨 비교하는 방식 |

## 헷갈리기 쉬운 짝

| A | B | 차이 |
|---|---|---|
| parameter | hyperparameter | 학습으로 바뀜 / 사람이 정함 |
| attention score | attention weight | softmax 전 / 후 |
| embedding (`nn.Embedding`) | hidden state | 문맥 없는 lookup / 블록을 거쳐 문맥이 반영된 값 |
| logits | probability | softmax 전 / 후 |
| token | token ID | 텍스트 조각 / 그 번호 |
| layer | block | GPT 문맥에서는 대개 같은 의미(`n_layers` = 블록 수) |
| stacking | recursion | 다른 가중치 블록을 쌓음 / 같은 함수를 자기 자신이 호출 |
| W_query 등 | Q·K·V | 학습으로 정해져 저장되는 가중치 / 입력마다 새로 계산되는 값 |
| LayerNorm | BatchNorm | token 하나의 feature 축으로 정규화 / batch 축으로 정규화 |
