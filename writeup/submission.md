# Jeremy AI x Gemma 4 — Access to Justice for All

## The Problem

80% of low-income Americans cannot afford a lawyer. Every year, millions of people face evictions, employment disputes, debt lawsuits, and civil rights violations with no legal help — because a lawyer costs $300+/hour. They file wrong forms. They miss deadlines. They lose cases they should have won.

The legal system was built for lawyers. **Jeremy was built for everyone else.**

## What Jeremy Does

Jeremy AI is a legal procedural guidance engine that walks self-represented litigants through the legal system step by step. It doesn't hallucinate legal advice — it routes users through an **80-phase deterministic system** backed by JSON rule databases covering **12 areas of law** across **13 U.S. jurisdictions**.

**Gemma 4 E4B** powers the natural language layer:
- **Intake:** Understands a user's situation in plain English
- **Fact Extraction:** Extracts structured legal facts (dates, entities, risk signals)
- **Guidance:** Explains procedures, deadlines, and next steps in plain English
- **Document Prep:** Helps draft Answers, EEOC Charges, Motions to Dismiss

The deterministic engine handles the law:
- **State Machine:** 12 case states with strict transitions
- **Rule Engine:** JSON rules defining prerequisites, deadlines, service requirements
- **Risk Scoring:** 0-100 composite score — custody detected = immediate attorney referral
- **Attorney Referral:** Off-ramp when the system reaches its limits

## Architecture

```
User speaks/types → Gemma 4 E4B (NLU/generation)
                          ↓
                    State Machine (routing)
                          ↓
                    Rule Engine (JSON — prerequisites, deadlines, forms)
                          ↓
                    Risk Engine (0-100 score)
                          ↓
              ┌───────────┴───────────┐
         Risk < 80                Risk >= 80
              ↓                       ↓
    Guidance + Documents      Attorney Referral
```

**Why this matters:** The LLM never decides the law. Rules are never generated — they're loaded from verified JSON databases. Every citation, every deadline, every prerequisite comes from the rule engine. Gemma 4 explains and translates. The engine decides.

## Training

Fine-tuned Gemma 4 E4B-it with QLoRA (rank 16, 4-bit quantization) on **5,196 legal Q&A pairs** collected from 5 free sources at $0 total cost:

| Source | Pairs | What |
|--------|-------|------|
| Synthetic (from rule JSONs) | 959 | Gold-standard procedural Q&A |
| law.stackexchange.com | 2,805 | Real legal questions from real people |
| SEC EDGAR | 202 | Real employment contracts |
| CourtListener API | 1,119 | Real case law extracts |
| Court self-help guides | 111 | Official court procedures |

Training ran on Kaggle free GPU (T4/P100) in under 2 hours.

## Real-World Impact

Jeremy has already been used in active litigation:
- **McDaniel v. City of New York** (Bronx County Supreme Court, 2026) — Section 1983 civil rights case against 6 NYPD officers. Jeremy helped draft the complaint, demand letter, and preservation notices.
- Multiple pro se litigants guided through EEOC filings, contract disputes, and eviction defense.

## What Gemma 4 Brings

Why Gemma 4 over other models?

1. **Size efficiency** — E4B runs on consumer hardware and mobile devices. Legal help shouldn't require a data center.
2. **128K context** — Full complaints, contracts, and case law fit in a single prompt.
3. **Open weights** — Self-hosted, no API costs. Access to justice shouldn't have a per-token fee.
4. **Instruction-tuned** — Gemma 4's instruction following maps perfectly to procedural legal guidance.

## Cost

| Component | Monthly Cost |
|-----------|-------------|
| Hosting (Oracle Free Tier) | $0 |
| Gemma 4 inference (self-hosted) | $0 |
| Training (Kaggle GPU) | $0 |
| Training data collection | $0 |
| Voice (ElevenLabs, optional) | $5 |
| **Total** | **$0 - $5** |

## Try It

[Live Demo Link — Kaggle Notebook / HuggingFace Space]

Ask Jeremy anything:
- "I was served with a lawsuit in New York. I have 20 days to respond. What do I do?"
- "My landlord is evicting me without proper notice in California."
- "How do I file an EEOC complaint for workplace discrimination?"

## Team

- **Israel "Ace" Burns** — Cornell Tech LLM '17, Builder
- **Esco Obong** — Collaborator
