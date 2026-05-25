# Fine-Tuning Mistral-7B on Financial Q&A with QLoRA

> Fine-tune a 7B parameter language model on a custom finance dataset using LoRA adapters and 4-bit quantization — all on a **free Google Colab T4 GPU**. Compare the fine-tuned model against the base model and a cloud API (Groq).

---

## What Problem Are We Solving?

Large language models like Mistral-7B are great at general knowledge, but they:
- Give **long, rambling answers** when you want short and precise ones
- Don't consistently use **domain-specific vocabulary**
- Can't be customized without expensive full retraining

**The question:** Can we take a 7B model, teach it to answer finance questions in a specific style, and do it entirely for free on a single GPU?

**The answer:** Yes — using QLoRA, which makes this possible in ~4GB of VRAM instead of the 80GB+ normally required.

---

## What We Built

A complete pipeline that:

1. Creates a **183-pair financial Q&A dataset** covering EBITDA, P/E ratios, DCF, WACC, bonds, options, and 150+ other finance concepts
2. **Fine-tunes Mistral-7B** using LoRA adapters — only 0.1% of model parameters are trained
3. Compresses the model from **14GB → 4GB** using 4-bit quantization (QLoRA)
4. **Evaluates** the fine-tuned model against the original base model and Groq's cloud API
5. Saves everything to Google Drive so work survives Colab session resets

---

## Results

| Model | Keyword Match | Avg Response Length | Format Consistency | Finance Term Density |
|-------|:---:|:---:|:---:|:---:|
| Base Mistral-7B | 50.7% | 117 words | 0% concise | 5.0% |
| **Fine-tuned (ours)** | **48.7%** | **34 words** | **100% concise** | **6.1%** |
| Groq API (Llama 3.1) | 49.0% | 101 words | 9% concise | 5.9% |

### Why the keyword score looks similar — but the fine-tuned model actually won

The keyword metric compared answers against short 25-word references. The base model gave 117-word essays, which paradoxically scored worse. The real wins for fine-tuning:

- **100% format consistency** — every answer was concise and on-point (base model: 0%)
- **22% more finance vocabulary** — domain language improved from 5.0% to 6.1%
- **Style transfer worked perfectly** — model learned to match the target answer length and structure

---

## How It Works — Simple Explanation

### The core idea: LoRA (Low-Rank Adaptation)

Instead of retraining all 7 billion weights of the model (which would cost hundreds of dollars and require massive GPUs), LoRA adds tiny "adapter" matrices to specific layers and only trains those.

```
Original model weights  →  FROZEN (never touched)
LoRA adapter matrices   →  TRAINED on our finance data
```

Mathematically, instead of updating a large weight matrix W directly, LoRA learns two small matrices A and B:

```
W_new = W_original + (A × B)
```

Where A and B are much smaller — `r=16` means the rank is 16, so instead of updating millions of values, you update thousands. This is why only **0.1% of parameters** were trained.

### QLoRA — fitting a 7B model into 4GB

Mistral-7B in normal (float16) precision = **~14GB VRAM**. A free T4 GPU only has 14.5GB total, leaving almost nothing for training.

QLoRA compresses the model weights from 16-bit to **4-bit precision** (nf4 format):
- 14GB model → ~4GB model
- Quality loss is minimal because nf4 is designed for neural networks
- Training still happens in float16 — only storage is in 4-bit

```
Without QLoRA:  Need 80GB+ GPU  →  Costs hundreds of dollars
With QLoRA:     Need 4GB GPU    →  Free on Google Colab T4
```

### The training format (instruction tuning)

Every training example was formatted as:

```
<s>[INST] What is EBITDA? [/INST] EBITDA stands for Earnings Before 
Interest, Taxes, Depreciation and Amortization. It measures a company's 
core operational profitability. </s>
```

The model learned: when it sees `[INST] ... [/INST]`, produce a concise, accurate financial answer in that exact style.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                     │
│                                                         │
│  train_pairs.json (160 pairs)                           │
│         ↓                                               │
│  Format as [INST] prompts                               │
│         ↓                                               │
│  Mistral-7B (4-bit quantized) + LoRA adapters           │
│         ↓                                               │
│  SFTTrainer — 3 epochs, loss: 3.1 → 0.24               │
│         ↓                                               │
│  Save adapter_config.json to Google Drive               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   EVALUATION PIPELINE                    │
│                                                         │
│  test_pairs.json (23 pairs)                             │
│         ↓                                    ↓          │
│  Fine-tuned model              Base model    Groq API   │
│         ↓                           ↓           ↓       │
│              Compare keyword match, length, style        │
│                         ↓                               │
│                  final_results.json                      │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Mistral-7B-Instruct-v0.3** | Base model — open source, no HuggingFace token needed |
| **Unsloth** | 2x faster training, handles CUDA version automatically |
| **QLoRA / BitsAndBytes** | 4-bit quantization — shrinks model from 14GB to 4GB |
| **LoRA / PEFT** | Adds tiny trainable layers without touching original weights |
| **SFTTrainer / TRL** | Clean training loop with gradient accumulation |
| **Google Colab T4** | Free GPU — 14.5GB VRAM, CUDA 12.8 |
| **Groq API** | Cloud inference for comparison (free tier) |
| **Google Drive** | Persists dataset and adapter between Colab sessions |

### LoRA Configuration

```python
LoraConfig(
    r=16,                           # Rank — controls adapter size
    lora_alpha=32,                  # Scaling factor (alpha/r = 2)
    target_modules=["q_proj",       # Which attention layers to adapt
                    "v_proj", 
                    "k_proj", 
                    "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Training Configuration

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch = 8
    learning_rate=2e-4,
    optim="adamw_8bit",             # 8-bit optimizer saves more VRAM
    fp16=True,                      # Mixed precision training
)
```

---

## Dataset

183 finance Q&A pairs covering:

- **Valuation:** EBITDA, P/E ratio, DCF, NAV, EV/EBITDA, P/B ratio
- **Financial statements:** Balance sheet, income statement, cash flow
- **Instruments:** Bonds, options, futures, ETFs, mutual funds
- **Corporate finance:** WACC, CapEx, working capital, leverage, LBO
- **Markets:** Bull/bear markets, yield curve, monetary policy, QE
- **Risk:** Beta, Sharpe ratio, systematic vs unsystematic risk, VaR
- **Concepts:** Compound interest, time value of money, diversification

Split: **160 training / 23 test** (held-out, never seen during training)

---

## How to Run

### Prerequisites
- Google Colab (free tier works, T4 GPU required)
- Google Drive (for saving checkpoints)
- Groq API key — free at [console.groq.com](https://console.groq.com)
- No HuggingFace token needed (Mistral is open access)

### Notebook Structure

```
CELL 0  — Install dependencies (unsloth, trl, groq)
CELL 1  — Verify GPU + mount Google Drive
CELL 2+3 — Load model → train LoRA → save adapter → fine-tuned inference
CELL 4  — Base model inference + Groq API + final evaluation
```

### Critical rules
1. Run cells **in order** — never skip
2. **Never restart** the runtime between cells
3. Cell 2+3 is one combined cell — training and inference must be in the same session

### Key file paths (Google Drive)
```
finance_lora_project/
├── train_pairs.json      ← full dataset (183 pairs)
├── adapter/              ← saved LoRA weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── ft_outputs.json       ← fine-tuned model answers
├── base_outputs.json     ← base model answers  
└── final_results.json    ← all results with scores
```

---

## Key Concepts for Interviews

### When to fine-tune vs RAG vs prompting

| Approach | Best when... | Limitation |
|----------|-------------|------------|
| **Fine-tuning** | Consistent format needed, domain tone, high volume, low latency | Doesn't update knowledge |
| **RAG** | Facts change over time, knowledge retrieval needed | Higher latency, more complex |
| **Prompting** | One-off tasks, need GPT-4 quality | Expensive at scale, no style control |

### Why fine-tuning didn't add knowledge but did add style

Mistral already knew finance from pretraining. Fine-tuning taught it **how to answer** — concisely, in the right format, with domain vocabulary. This is what supervised fine-tuning (SFT) is designed for: behavior shaping, not knowledge injection.

### Catastrophic forgetting risk

Because we only trained LoRA adapters (0.1% of weights), the base model's general knowledge was preserved. Full fine-tuning would have risked the model forgetting non-finance knowledge — LoRA avoids this by design.

---

## Challenges Survived

This project required navigating **16 dependency conflicts** across:

- `bitsandbytes` CUDA binary mismatches (cuda130.so not found)
- `numpy` binary incompatibility (dtype size changed)
- `transformers` version requiring conflicting `bitsandbytes` versions
- `trl`/`peft`/`torchao` cascading version locks
- HuggingFace gated model access (solved by switching to Mistral)
- HuggingFace outage mid-session
- OOM errors from loading two models simultaneously
- Colab runtime resets wiping all pip installs
- Groq model deprecation (`llama3-8b-8192` → `llama-3.1-8b-instant`)

**Final working solution:** Use [Unsloth](https://github.com/unslothai/unsloth) which ships pre-compiled CUDA binaries and pre-quantized models, bypassing all bitsandbytes compilation issues.

---

## Interview Hook

> *"Fine-tuned Mistral-7B with QLoRA on a free T4 GPU. Only 0.1% of parameters were trained using LoRA adapters. The fine-tuned model achieved 100% format consistency versus 0% for the base model, and 22% higher finance term density. Keyword scores were similar across all three models at ~49-51%, but response length analysis revealed the fine-tuned model learned the target style precisely — 34 words average versus 117 for the base. Survived 16 environment conflicts across CUDA, numpy, bitsandbytes, and transformers — the real learning was as much about ML infrastructure as the training itself."*

---

## References

- [LoRA paper — Hu et al. 2021](https://arxiv.org/abs/2106.09685)
- [QLoRA paper — Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
