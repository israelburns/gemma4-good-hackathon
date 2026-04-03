# Jeremy AI x Gemma 4 — Access to Justice for All

**Gemma 4 Good Hackathon Submission**
Kaggle: https://www.kaggle.com/competitions/gemma-4-good-hackathon

## The Problem

80% of low-income Americans cannot afford a lawyer (Legal Services Corporation).
Every year, millions navigate the legal system alone — filing wrong forms, missing deadlines,
losing cases they should have won — because legal help costs $300+/hour.

## The Solution

**Jeremy AI** is a deterministic legal guidance engine powered by **Gemma 4 E4B**,
purpose-built for self-represented litigants. It doesn't hallucinate legal advice —
it routes users through an 80-phase procedural system backed by JSON rule databases
covering 12 areas of law across 13 U.S. jurisdictions.

Gemma 4 handles natural language understanding (intake, fact extraction, plain-English
guidance) while the deterministic engine handles the law (deadlines, prerequisites,
document generation, risk scoring).

## Architecture

```
User Input → Gemma 4 E4B (NLU) → State Machine → Rule Engine → Risk Scoring
                                                      ↓
                                              Document Generation
                                              Attorney Referral (if risk > 80)
```

**5-Layer Stack:**
1. State Machine — 12 deterministic case states
2. Rule Engine — JSON rules (prerequisites, deadlines, service rules)
3. Risk Scoring — 0-100 composite, custody = instant attorney referral
4. Deliverables — Answer, EEOC Charge, Motion to Dismiss, etc.
5. Attorney Referral — off-ramp when AI reaches its limits

## What Makes This Different

- **Deterministic legal logic** — LLM explains, rules decide. Zero hallucinated citations.
- **Real cases served** — McDaniel v. City of New York (Section 1983, active litigation)
- **$0 infrastructure** — Oracle Free Tier + Kaggle GPU + free APIs
- **80 phases built** — not a prototype, a production system

## Training Data

5,196 pairs across 5 free sources ($0 total):
- Synthetic (rule-based): 959 pairs
- StackExchange (law.stackexchange.com): 2,805 pairs
- SEC EDGAR (real employment contracts): 202 pairs
- CourtListener (real case law): 1,119 pairs
- Court self-help guides: 111 pairs

## Files

```
├── notebooks/
│   ├── gemma4_finetune.py      # Kaggle fine-tuning notebook (Gemma 4 E4B + LoRA)
│   └── gemma4_demo.py          # Inference demo notebook
├── writeup/
│   └── submission.md           # Kaggle writeup draft
├── video/
│   └── script.md               # 2-min video script
└── README.md
```

## Linked Repos (dedup — no code copied)

- **Pro Se Network**: github.com/israelburns/pro-se-network (80-phase engine, training data, frontend)
- **Jeremy v1 Model**: huggingface.co/israelburns/jeremy-v1 (Phi-3 fine-tune, being replaced by Gemma 4)
- **Training Data**: huggingface.co/israelburns/jeremy-training-data

## Team

- **Israel "Ace" Burns** — Builder, Cornell Tech LLM '17
- **Esco Obong** — Collaborator

## Deadline

May 18, 2026 at 7:59 PM EDT
