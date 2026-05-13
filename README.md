# Jeremy AI x Gemma 4 — Access to Justice for All

[![Kaggle](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Model](https://img.shields.io/badge/HuggingFace-israelburns%2Fjeremy--gemma4-FFD21E?logo=huggingface)](https://huggingface.co/israelburns/jeremy-gemma4)
[![Dataset](https://img.shields.io/badge/Dataset-israelburns%2Fjeremy--training--data-FFD21E?logo=huggingface)](https://huggingface.co/datasets/israelburns/jeremy-training-data)
[![Demo](https://img.shields.io/badge/Live%20Demo-prosenetwork.org%2Fdemo-00C853?logo=google-chrome)](https://prosenetwork.org/demo)
[![Notebook](https://img.shields.io/badge/Kaggle%20Notebook-jeremy--ai--x--gemma--4--fine--tune-20BEFF?logo=kaggle)](https://www.kaggle.com/code/israelburns/jeremy-ai-x-gemma-4-fine-tune)

---

## The Problem

80% of low-income Americans cannot afford a lawyer. In Kenya, India, Nigeria, and across the developing world, the justice gap is not a statistic — it is a daily reality for millions facing evictions, wage theft, wrongful terminations, and civil rights violations with no one in their corner.

A lawyer costs $300+/hour in the U.S. In ***, a single court filing can cost more than a month's income. People file wrong forms. They miss deadlines. They lose cases they should have won — not because they were wrong, but because they couldn't navigate a system built for lawyers.

**Jeremy was built for everyone else.**

---

## The Solution

**Jeremy AI** is a legal procedural guidance engine powered by **Gemma 4 E4B**, purpose-built for self-represented litigants. It doesn't hallucinate legal advice — it routes users through an **80-phase deterministic system** backed by JSON rule databases covering **12 areas of law** across **13 U.S. jurisdictions**.

Gemma 4 handles the natural language layer. The rule engine handles the law.

### Architecture

```
User Input → Gemma 4 E4B (NLU + generation)
                    ↓
             State Machine (12 case states)
                    ↓
             Rule Engine (JSON — deadlines, prerequisites, forms)
                    ↓
             Risk Engine (0–100 composite score)
                    ↓
        ┌───────────┴───────────┐
   Risk < 80               Risk ≥ 80
        ↓                       ↓
Guidance + Documents     Attorney Referral
```

**Why this matters:** The LLM never decides the law. Rules are never generated — they're loaded from verified JSON databases. Gemma 4 explains and translates. The engine decides.

---

## Fine-Tuning

### Model
- **Base:** `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` (Gemma 4 E4B instruction-tuned)
- **Method:** QLoRA — rank 16, alpha 32, 4-bit quantization
- **Adapter:** [huggingface.co/israelburns/jeremy-gemma4](https://huggingface.co/israelburns/jeremy-gemma4)

### Training Specs
| Parameter | Value |
|-----------|-------|
| GPU | Kaggle T4 (15.6GB VRAM, sm_75, fp16) |
| Steps | 927 (3 epochs × 309 steps) |
| Effective batch size | 16 (batch 1 × grad accum 16) |
| Seq length | 1024 |
| Learning rate | 2e-4 (cosine decay) |
| Optimizer | adamw_8bit |
| Time | 5h 18m |
| Cost | $0 |

### Training Data — 5,196 pairs ($0 total)

| Source | Pairs | What |
|--------|-------|------|
| Synthetic (rule JSONs) | 959 | Gold-standard procedural Q&A |
| law.stackexchange.com | 2,805 | Real legal questions from real people |
| SEC EDGAR | 202 | Real employment contracts |
| CourtListener API | 1,119 | Real case law extracts |
| Court self-help guides | 111 | Official court procedures |

Dataset: [huggingface.co/datasets/israelburns/jeremy-training-data](https://huggingface.co/datasets/israelburns/jeremy-training-data)

---

## Why Gemma 4

1. **Open weights — the mission requirement.** No API. No per-token cost. No vendor lock-in. Any legal aid org anywhere can self-host this.
2. **E4B efficiency** — runs on consumer hardware. Legal help shouldn't require a data center.
3. **128K context** — full complaints, contracts, and case law fit in a single prompt.
4. **Instruction-tuned** — Gemma 4's instruction following maps directly to procedural guidance.
5. **Global deployability** — fine-tune for any jurisdiction, any language.

---

## Cost

| Component | Monthly Cost |
|-----------|-------------|
| Hosting (Oracle Free Tier) | $0 |
| Gemma 4 inference (self-hosted) | $0 |
| Training (Kaggle GPU) | $0 |
| Training data | $0 |
| Voice (Kokoro TTS, self-hosted) | $0 |
| **Total** | **$0** |

---

## Real-World Impact

Jeremy has already been used in active litigation:

- **McDaniel v. City of New York** (Bronx County Supreme Court, 2026) — Section 1983 civil rights case against 6 NYPD officers. Jeremy helped draft the complaint, demand letter, and preservation notices.
- Multiple pro se litigants guided through EEOC filings, contract disputes, and eviction defense across New York.

---

## A Global Build

*******, based in ***, Kenya, tested the platform and helped frame the story from the outside — the perspective of someone who lives the justice gap, not just reads about it. His input shapes a core requirement: the architecture must be portable, free, and self-hostable by communities that have never had access to $300/hour counsel. The choice of open weights is partly an answer to that.

---

## Try It

- **Live Demo:** [prosenetwork.org/demo](https://prosenetwork.org/demo)
- **Model:** [huggingface.co/israelburns/jeremy-gemma4](https://huggingface.co/israelburns/jeremy-gemma4)
- **Notebook:** [kaggle.com/code/israelburns/jeremy-ai-x-gemma-4-fine-tune](https://www.kaggle.com/code/israelburns/jeremy-ai-x-gemma-4-fine-tune)

Ask Jeremy anything:
> "I was served with a lawsuit in New York. I have 20 days to respond. What do I do?"
> "My landlord is evicting me without proper notice in California."
> "How do I file an EEOC complaint for workplace discrimination?"

---

## Files

```
├── notebooks/
│   └── gemma4_finetune_unsloth.py   # Kaggle fine-tuning script (Gemma 4 E4B + QLoRA + Unsloth)
├── writeup/
│   └── submission.md                # Full Kaggle competition writeup
├── video/
│   └── script.md                    # 2-min demo video script
└── README.md
```

---

## Contributor

**Israel "Ace" Burns** — Cornell Tech LLM '17 · Co-Founder, Pro Se Network · New York
