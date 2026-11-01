# GPT-2 from Scratch

LLM을 밑바닥부터 직접 구현하며 GPT-2(124M) 구조를 학습하는 저장소입니다.
토큰화부터 어텐션, 트랜스포머 블록, GPT 모델까지 단계별로 코드를 작성하고,
학습 중 생긴 질문과 답변은 [`docs/`](docs/)에 정리합니다.

## 학습 로드맵

| 단계 | 폴더 | 내용 | 상태 |
|---|---|---|---|
| 1 | [`tokenizer/`](tokenizer/) | 단순 토크나이저, BPE(`tiktoken`), 슬라이딩 윈도우 데이터셋, 토큰/위치 임베딩 | ✅ |
| 2 | [`attention/`](attention/) | 기본 어텐션 → scaled dot-product → self-attention → causal → multi-head | ✅ |
| 3 | [`transformer/`](transformer/) | GPT 설정(124M), 더미 GPT 모델, LayerNorm, GELU/FeedForward, 트랜스포머 블록 | 🚧 |
| 4 | — | GPT 모델 완성 및 텍스트 생성 | ⬜ |
| 5 | — | 사전 훈련(pretraining), 손실 계산, 사전 훈련 가중치 로드 | ⬜ |
| 6 | — | 미세 튜닝(분류 / 지시 따르기) | ⬜ |

## 디렉터리 구조

```
gpt-2/
├── tokenizer/     # 토큰화와 임베딩 (the-verdict.txt 데이터 포함)
├── attention/     # 어텐션 메커니즘 구현
├── transformer/   # GPT 모델 구성 요소
├── docs/          # 학습 Q&A 노트
├── AGENTS.md      # AI 에이전트용 작업 지침
└── requirements.txt
```

## 환경 설정

```bash
# 0. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 1. 패키지 설치
pip install torch tiktoken

# 2. 설치된 패키지 목록을 파일로 저장
pip freeze > requirements.txt

# 3. 저장된 명세서로 동일하게 패키지 일괄 설치
pip install -r requirements.txt
```

## 실행 예시

각 파일은 단독 실행 가능한 학습 스크립트입니다. 저장소 루트에서 실행합니다.

```bash
python tokenizer/sliding_window.py
python attention/01_basic_attention.py
python transformer/layer_norm.py
```

## 학습 Q&A (`docs/`)

공부하면서 생긴 질문과 답변을 단계별로 정리합니다.

- `docs/tokenizer/`, `docs/attention/`, `docs/transformer/`, `docs/general/`
- 파일명 규칙: `NN-주제-키워드.md` (예: `docs/attention/01-why-scale-by-sqrt-dk.md`)
- 각 문서는 **질문 → 짧은 답 → 자세한 설명 → 코드 예시 → 더 알아볼 것** 순서로 작성
- 전체 목차: [`docs/README.md`](docs/README.md)

자세한 작성 규칙은 [`AGENTS.md`](AGENTS.md)를 참고하세요.
