# GPT는 분할 정복 / 재귀 구조인가?

- 날짜: 2026-11-01
- 관련 코드: `transformer/dummy_gpt_model.py`, `attention/multi_head_attetion.py`
- 키워드: stacking, function composition, recursion, divide and conquer, weight sharing

## 질문
GPT는 비슷한 구조가 계속 반복되는데, 분할 정복 구조인가? 아니면 재귀 구조라는 표현이 맞는가?

## 짧은 답
둘 다 아니다. **구조는 같고 가중치는 다른 블록을 차례로 쌓은 것(stacking)**이다.
수식으로는 함수 합성이다: $h_{12} = f_{12}(f_{11}(\cdots f_1(h_0)))$

## 비교

| 표현 | 맞는가 | 이유 |
|---|---|---|
| 분할 정복 | ❌ | 문제를 나누지 않는다. 나누고 합치는 건 MHA 내부의 head 분할뿐이다 |
| 재귀 | ❌ | 자기 자신을 호출하지 않고, 블록 12개의 가중치가 서로 다르다 |
| stacking (함수 합성) | ✅ | 같은 설계의 블록 12개를 앞에서 뒤로 한 번씩 통과한다 |

```python
# 서로 다른 객체(가중치) 12개를 만든다. 같은 함수를 다시 부르는 게 아니다
self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(12)])
```

반복해서 쌓을 수 있는 이유는 블록의 입력과 출력 shape가 `(batch, seq_len, 768)`로 같기 때문이다.

## "반복 적용"에 가까운 모델
같은 가중치를 여러 번 다시 쓰면 $h_{t+1} = f(h_t)$ 형태가 되어 재귀에 가까워진다.

- **RNN**: 같은 셀을 시간 순서대로 반복 적용한다(recurrent).
- **Universal Transformer / ALBERT**: 블록 하나의 가중치를 모든 층이 공유한다.
- **GPT-2**: 공유하지 않는다. 그래서 층마다 다른 역할(얕은 패턴 → 추상적 의미)을 배울 수 있다.

## 더 알아볼 것
- 블록 하나의 파라미터는 약 7M이고, 12개를 합하면 약 85M이다(124M 중 나머지는 대부분 임베딩).
- 전체 구조: [architecture.md](../architecture.md) · 용어: [glossary.md](../glossary.md)
