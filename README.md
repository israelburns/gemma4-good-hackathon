# Jeremy AI x Gemma 4 — Access to Justice for All

**Gemma 4 Good Hackathon Submission**
Kaggle: https://www.kaggle.com/competitions/gemma-4-good-hackathon

---

## The Mission

Jeremy AI is not a commercial product. It is a **nonprofit infrastructure project.**

80% of low-income Americans cannot afford a lawyer. The number is worse globally — in Kenya, India, Nigeria, and across the developing world, the justice gap is a daily reality for millions facing evictions, wage theft, and civil rights violations with no help.

The decision to build on Gemma 4 open weights is a mission requirement. Legal aid organizations, public defenders, and community legal clinics cannot afford subscription AI. They serve the people who need help most, with the least resources.

Gemma 4 makes it possible to run Jeremy anywhere in the world, on any hardware, at **$0 per conversation, forever.**

## The Solution

**Jeremy AI** is a deterministic legal guidance engine powered by **Gemma 4 E4B**, purpose-built for self-represented litigants. It doesn't hallucinate legal advice — it routes users through an 80-phase procedural system backed by JSON rule databases covering 12 areas of law across 13 U.S. jurisdictions.

Gemma 4 handles natural language (intake, fact extraction, plain-English guidance) while the deterministic engine handles the law (deadlines, prerequisites, document generation, risk scoring).

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

## Why Gemma 4

- **Open weights** — no API, no per-token cost, no vendor lock-in. Self-hostable by any org globally.
- **128K context** — full contracts and complaints fit in one prompt
- **E4B efficiency** — runs on consumer hardware, no data center needed
- **Portable** — fine-tune for any jurisdiction, any language

## Real-World Impact

- **McDaniel v. City of New York** (Bronx Supreme Court, 2026) — Section 1983 against 6 NYPD officers. Jeremy drafted the complaint, demand letter, and preservation notices.
- Multiple pro se litigants guided through EEOC filings, eviction defense, and contract disputes

## A Global Build

This project has a transatlantic dimension. *******, based in ***, Kenya, is our third collaborator and brings direct perspective on the justice gap outside the U.S. legal system. His contribution shapes the global deployability requirement: the architecture must be portable, free, and self-hostable by communities that have never had access to $300/hour counsel. The choice of open weights is partly an answer to that.

## Training Data

5,196 pairs across 5 free sources ($0 total):
- Synthetic (rule-based): 959 pairs
- StackExchange (law.stackexchange.com): 2,805 pairs
- SEC EDGAR (real employment contracts): 202 pairs
- CourtListener (real case law): 1,119 pairs
- Court self-help guides: 111 pairs

## Cost

| Component | Monthly Cost |
|-----------|-------------|
| Hosting (Oracle Free Tier) | $0 |
| Gemma 4 inference (self-hosted) | $0 |
| Training (Kaggle GPU) | $0 |
| Training data | $0 |
| Voice (optional) | $5 |
| **Total** | **$0–$5** |

## Files

```
├── notebooks/
│   ├── gemma4_finetune.py      # Kaggle fine-tuning script (Gemma 4 E4B + QLoRA)
│   └── gemma4_demo.py          # Inference demo
├── writeup/
│   └── submission.md           # Full Kaggle writeup
├── video/
│   └── script.md               # 2-min video script
└── README.md
```

## Links

- **Model:** [huggingface.co/israelburns/jeremy-gemma4](https://huggingface.co/israelburns/jeremy-gemma4)
- **Training Data:** [huggingface.co/datasets/israelburns/jeremy-training-data](https://huggingface.co/datasets/israelburns/jeremy-training-data)
- **Pro Se Network:** [prosenetwork.org](https://prosenetwork.org)

## Team

- **Israel "Ace" Burns** — Cornell Tech LLM '17, Builder, New York
- ******* — Collaborator & Global Perspective, ***, Kenya
- **Esco Obong** — Collaborator, New York

## Deadline

May 18, 2026 at 7:59 PM EDT
