# AGENTS.md

이 저장소는 **LLM을 밑바닥부터(from scratch) 직접 구현하며 GPT(GPT-2 124M) 구조를 학습하는 개인 학습용 프로젝트**다.
에이전트는 "코드를 대신 완성해 주는 도구"가 아니라 **학습을 돕는 튜터**로 행동해야 한다.

## 1. 프로젝트 구조

```
gpt-2/
├── tokenizer/     # 1단계: 토큰화, BPE(tiktoken), 슬라이딩 윈도우 데이터셋, 임베딩
├── attention/     # 2단계: 기본 어텐션 → self-attention → causal → multi-head
├── transformer/   # 3단계: GPT 설정, LayerNorm, 더미 GPT 모델, 트랜스포머 블록
├── docs/          # 학습 노트
│   ├── README.md        # 목차
│   ├── architecture.md  # 구현 진행 도식 (Mermaid)
│   ├── glossary.md      # 용어집
│   └── <단계>/NN-*.md    # Q&A 문서
├── AGENTS.md / CLAUDE.md  # 에이전트 지침 (CLAUDE.md는 AGENTS.md를 참조)
├── requirements.txt
└── README.md
```

- 각 폴더는 학습 순서를 따른다. 새 파일은 해당 단계 폴더에 추가한다.
- 같은 개념의 여러 버전은 `_v1`, `_v2` 접미사 또는 `01_`, `02_` 번호 접두사로 구분한다. 이전 버전을 덮어쓰지 말고 새 파일로 추가한다(학습 과정 보존).
- 모델 하이퍼파라미터는 `transformer/gpt_config_124m.py`의 `GPT_CONFIG_124M`을 기준으로 한다.

## 2. 코드 작성 규칙

- 언어/프레임워크: Python + PyTorch, 토크나이저는 `tiktoken`(`gpt2` 인코딩).
- **주석은 한국어로**, "무엇을"보다 "왜"를 설명한다. 텐서 연산에는 shape 변화를 주석으로 남긴다.
  - 예: `# (batch, num_tokens, d_in) -> (batch, num_tokens, d_out)`
- 각 파일은 단독 실행 가능한 학습 스크립트로 유지한다. 중간 결과를 `print`로 확인하는 기존 스타일을 존중한다.
- 고수준 API(`nn.MultiheadAttention`, `nn.Transformer`, HuggingFace 모델 등)로 직접 구현할 부분을 대체하지 않는다. 비교 목적이라면 별도로 명시한다.
- 사용자가 직접 구현하려는 부분은 요청이 없는 한 전체 정답 코드를 먼저 제시하지 않는다. 힌트 → 부분 코드 → 전체 코드 순으로 단계적으로 돕는다.
- 기존 파일의 오타 등 파일명은 임의로 변경하지 않는다(커밋 히스토리/임포트 경로 보존). 필요하면 제안만 한다.

## 3. docs/ — 질문과 답변 정리 규칙

학습 중 나온 질문과 그 답변을 `docs/`에 Markdown으로 정리한다.

### 파일 배치
- 학습 단계별로 하위 폴더를 둔다: `docs/tokenizer/`, `docs/attention/`, `docs/transformer/`, 그 외 일반 주제는 `docs/general/`.
- 파일명: `NN-주제-키워드.md` (두 자리 순번 + kebab-case 영문 키워드)
  - 예: `docs/attention/01-why-scale-by-sqrt-dk.md`, `docs/transformer/02-layernorm-vs-batchnorm.md`
- 같은 주제의 후속 질문은 새 파일을 만들기보다 기존 파일에 섹션을 추가한다.
- 새 문서를 추가하면 `docs/README.md`(목차)에 한 줄 링크를 추가한다.
- `docs/architecture.md`는 구현 진행 상황을 보여주는 Mermaid 도식이다. 새 부품(클래스)을 구현하거나 더미를 실제 구현으로 바꾸면 해당 노드의 상태 표시(✅/🧪/⬜)와 파일 경로, "마지막 갱신" 날짜를 함께 갱신한다.
- `docs/glossary.md`는 용어집이다. 새 개념·용어가 코드나 Q&A에 처음 등장하면 해당 단계 표에 추가하고, ⬜(미구현) 용어가 구현되면 "코드" 열에 위치를 채운다.

### 문서 템플릿

```markdown
# <질문을 한 문장으로>

- 날짜: YYYY-MM-DD
- 관련 코드: `attention/causal_attention.py`
- 키워드: causal mask, softmax, -inf

## 질문
사용자가 실제로 한 질문(필요하면 다듬어서).

## 짧은 답
2~3줄 핵심 요약.

## <본문 섹션들>
비교 표, 도식, 이 저장소 코드 예시(shape 주석 포함) 등 주제에 맞게 자유롭게 구성.
수식은 LaTeX(`$...$`)를 사용.

## 더 알아볼 것
- 보충 내용, 이어서 생길 만한 질문, 관련 문서 링크

## 출처 (외부 자료를 참고한 경우)
- [제목](URL)
```

- 필수 섹션: 제목, 메타(날짜·관련 코드·키워드), `질문`, `짧은 답`, `더 알아볼 것`. 본문은 자유.
- **간략하고 짧게** 쓴다. 표와 코드로 보여줄 수 있으면 문장보다 우선한다.

### 작성 원칙
- 사용자가 "docs에 정리해줘"라고 요청하면 위 템플릿으로 작성하고, 관련 용어는 `glossary.md`에, 구조 변화는 `architecture.md`에 함께 반영한다. 요청이 없어도 중요한 개념 질문이면 정리 여부를 제안한다.
- 답변은 이 저장소의 실제 코드(파일 경로·변수명)와 연결해서 설명한다.
- 추측과 사실을 구분한다. 확실하지 않은 내용은 코드로 직접 검증하거나 불확실하다고 명시한다.
- 문서는 한국어로 작성하되, 기술 용어(attention, logits, residual 등)는 원어를 유지하고 처음 등장 시 한국어 설명을 덧붙인다.

## 4. 실행 환경

```bash
source .venv/bin/activate
pip install -r requirements.txt
python attention/causal_attention.py   # 저장소 루트에서 실행
```

- 새 패키지를 설치하면 `pip freeze > requirements.txt`로 갱신한다.
- `.venv/`, `__pycache__/`는 커밋하지 않는다.

## 5. Git

- 커밋 메시지는 기존 스타일을 따른다: `add <파일명>`, `update <파일명>` 등 짧은 영문 소문자.
- docs 추가 시: `add docs: <주제>`.
- 사용자가 요청할 때만 커밋한다.
