# Semantic dense 임베딩(KURE)과 지금 만드는 GPT는 무엇이 다른가?

- 날짜: 2026-11-01
- 관련 코드: `tokenizer/my_embedding.py`, `attention/causal_attention.py`, `transformer/dummy_gpt_model.py`
- 키워드: encoder-only, decoder-only, hidden state, pooling, contrastive learning, KURE

## 질문
Semantic dense 임베딩은 인코더 구조만 쓰는 것인가?
임베딩한 뒤 어텐션까지 처리한 결과가 우리가 실제로 쓰는 "임베딩"인가?

## 짧은 답
- 대체로 맞다. 대표 모델(KURE-v1 등)은 encoder-only다. 다만 decoder LLM을 바꿔 임베딩 모델로 쓰는 경우도 있어서 필수 조건은 아니다.
- 검색에 쓰는 임베딩 = **token embedding → 트랜스포머 블록 N개(attention + FFN) → pooling으로 벡터 1개**. 여기에 **contrastive 학습**을 해야 의미 유사도가 맞춰진다.
- `nn.Embedding`만 통과한 벡터는 문맥이 없는 lookup 값일 뿐이다.

## 동작 비교

```
[KURE-v1]  문장 → 토큰 → nn.Embedding + pos → Encoder 블록 × 24 (양방향 attention)
                → [CLS] 위치 hidden state 1개 → 정규화 → 1024차원 벡터  (끝)

[우리 GPT] 문장 → 토큰 → nn.Embedding + pos → Decoder 블록 × 12 (causal attention)
                → 모든 위치 hidden state(768) → out_head → 위치마다 logits(50257) → 다음 토큰
```

| | KURE-v1 | 우리 GPT (124M) |
|---|---|---|
| 베이스 | BAAI/bge-m3 (XLM-RoBERTa 인코더) | GPT-2 구조 |
| 어텐션 | 양방향 (앞뒤 모두 봄) | causal (앞쪽만 봄) |
| 마지막 출력 | 문장당 벡터 1개 (1024차원) | 토큰마다 vocab logits |
| 학습 목표 | contrastive: 질문-정답 문서는 가깝게, hard negative는 멀게 | 다음 토큰 예측 (cross-entropy) |
| 최대 길이 | 8192 토큰 | 1024 토큰 |
| 용도 | 검색, RAG retrieval, 유사도 | 텍스트 생성 |

**공통점**: 앞부분(토큰화 → embedding → attention 블록)은 같은 원리다. 차이는 **마스크 방향, 마지막 출력을 어떻게 쓰는지(pooling vs logits), 학습 목표** 세 가지다.

## Hidden state란?
모델 **중간 단계에서 토큰마다 나오는 벡터**다. 입력(토큰 ID)도 출력(logits)도 아닌, 모델 안에서만 다뤄지기 때문에 hidden(숨겨진)이라고 부른다.

`transformer/dummy_gpt_model.py`의 `forward`에서 보면:

```python
x = tok_embeds + pos_embeds   # (batch, seq_len, 768)  ← 입력 임베딩: 문맥 없음
x = self.trf_blocks(x)        # (batch, seq_len, 768)  ← 블록마다 hidden state가 갱신됨
x = self.final_norm(x)        # (batch, seq_len, 768)  ← 최종 hidden state
logits = self.out_head(x)     # (batch, seq_len, 50257) ← 여기서부터는 hidden state가 아님
```

- **shape는 처음부터 끝까지 `(batch, seq_len, emb_dim)`으로 같다.** 블록을 지날수록 값만 바뀌면서 주변 토큰의 문맥이 점점 섞여 들어간다.
- 같은 "bank"라도 입력 임베딩은 같지만, 블록을 통과한 hidden state는 문맥에 따라 달라진다.
- **최종 hidden state를 쓰는 방식이 모델마다 다르다.**
  - GPT: 토큰마다 `out_head`에 넣어 다음 토큰 logits를 만든다.
  - KURE-v1: `[CLS]` 위치의 hidden state 1개를 꺼내 문장 임베딩으로 쓴다.

## 더 알아볼 것
- KURE-v2는 문장당 벡터 1개가 아니라 토큰마다 벡터를 남기는 ColBERT(late-interaction, multi-vector) 방식으로 소개되어 있다. 그래서 일반 dense 임베딩(v1)과는 성격이 다르다.
- KURE-v3는 2026-11-01 기준 공개 자료에서 확인하지 못했다.
- 우리 GPT로 임베딩을 만들려면 마지막 토큰의 hidden state(last-token pooling)를 쓰고, contrastive 학습을 추가해야 한다.
- 용어 정리: [glossary.md](../glossary.md) · 전체 구조: [architecture.md](../architecture.md)

## 출처
- [nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1)
- [nlpai-lab/KURE-v2](https://huggingface.co/nlpai-lab/KURE-v2)
