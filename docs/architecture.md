# GPT-2 구현 진행 도식

- 마지막 갱신: 2026-11-01
- 범례: ✅ 구현 완료 · 🧪 더미(자리만 있음) · ⬜ 미구현

## 1. 전체 흐름

같은 구조의 Transformer Block을 `n_layers`(12)번 **반복해서 쌓는** 구조다.
블록의 입력과 출력 shape가 `(batch, seq_len, 768)`로 같기 때문에 반복할 수 있다.

```mermaid
flowchart TB
    T["텍스트"] --> TK["✅ Tokenizer<br/>tiktoken gpt2<br/>tokenizer/my_token.py"]
    TK --> DS["✅ GPTDatasetV1 / DataLoader<br/>슬라이딩 윈도우 (input, target)<br/>tokenizer/gpt_dataset_v1.py"]
    DS -->|"token IDs (batch, seq_len)"| TE["✅ Token Embedding<br/>nn.Embedding(50257, 768)"]
    DS --> PE["✅ Positional Embedding<br/>nn.Embedding(1024, 768)"]
    TE --> ADD(("+"))
    PE --> ADD
    ADD --> DROP["✅ Dropout"]
    DROP -->|"(batch, seq_len, 768)"| B1

    subgraph STACK["Transformer Block × 12 — 같은 구조 반복"]
        B1["🧪 Block 1"] --> B2["🧪 Block 2"] --> BD["⋯"] --> B12["🧪 Block 12"]
    end

    B12 -->|"최종 hidden state<br/>(batch, seq_len, 768)"| FN["✅ Final LayerNorm<br/>transformer/layer_norm.py"]
    FN --> OH["✅ out_head<br/>Linear(768 → 50257)"]
    OH -->|"logits (batch, seq_len, 50257)"| GEN["⬜ 텍스트 생성<br/>generate_text_simple"]

    CFG[/"✅ GPT_CONFIG_124M<br/>transformer/gpt_config_124m.py"/] -.-> STACK
```

> `DummyGPTModel`(`transformer/dummy_gpt_model.py`)에 위 뼈대가 연결되어 있다. Block은 아직 입력을 그대로 반환하는 더미다.

## 2. Transformer Block 내부

```mermaid
flowchart TB
    IN["x (batch, seq_len, 768)"] --> LN1["✅ LayerNorm 1"]
    LN1 --> MHA["✅ Masked Multi-Head Attention<br/>attention/multi_head_attetion.py"]
    MHA --> D1["⬜ Dropout"]
    D1 --> R1(("+"))
    IN -->|"⬜ shortcut (residual)"| R1

    R1 --> LN2["✅ LayerNorm 2"]
    LN2 --> FF["⬜ FeedForward<br/>Linear(768→3072) → GELU → Linear(3072→768)"]
    FF --> D2["⬜ Dropout"]
    D2 --> R2(("+"))
    R1 -->|"⬜ shortcut (residual)"| R2

    R2 --> OUT["x (batch, seq_len, 768)"]
```

> 부품(LayerNorm, MultiHeadAttention)은 만들어졌지만 `TransformerBlock` 클래스로 조립되지는 않았다.

## 3. Multi-Head Attention 내부

헤드 12개로 **나누어 병렬로 계산한 뒤 합친다**. 블록 반복과는 다른 종류의 구조다.

```mermaid
flowchart LR
    X["x (b, n, 768)"] --> QKV["✅ W_q, W_k, W_v<br/>Linear(768→768)"]
    QKV --> SPLIT["✅ 헤드 분할<br/>(b, 12, n, 64)"]
    SPLIT --> ATT["✅ QKᵀ / √64<br/>+ causal mask<br/>→ softmax → dropout"]
    ATT --> V["✅ × V"]
    V --> CAT["✅ 헤드 합치기<br/>(b, n, 768)"]
    CAT --> PROJ["✅ out_proj<br/>Linear(768→768)"]
```

## 4. 학습 경로 (attention/)

```mermaid
flowchart LR
    A1["01_basic_attention"] --> A2["02_scaled_dot_product"] --> A3["SelfAttention_v1/v2"]
    A3 --> A4["CausalAttention"] --> A5["MultiHeadAttentionWrapper<br/>(헤드 여러 개 단순 연결)"]
    A5 --> A6["MultiHeadAttention<br/>(가중치 분할, 효율적)"]
```

## 갱신 방법
새 부품을 구현하면 해당 노드의 ⬜/🧪를 ✅로 바꾸고 파일 경로를 적는다. 맨 위의 "마지막 갱신" 날짜도 바꾼다.
