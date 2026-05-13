# Jeremy AI x Gemma 4 — Access to Justice for All

## The Problem

80% of low-income Americans cannot afford a lawyer. Across the developing world, the justice gap is not a statistic. It is a daily reality for millions of people facing evictions, wage theft, wrongful terminations, and civil rights violations with no one in their corner.

A lawyer costs $300+/hour in the U.S. People file wrong forms. They miss deadlines. They lose cases they should have won — not because they were wrong, but because they couldn't navigate a system built for lawyers.

The legal system was built for lawyers. **Jeremy was built for everyone else.**

## The Mission — Built to Be Free

Jeremy AI is not a commercial product. It is a **nonprofit infrastructure project.**

The decision to build on Gemma 4 open weights is not a technical preference — it is a mission requirement. When access to justice is the goal, a per-token API bill is not acceptable. Legal aid organizations, public defenders, courthouse self-help centers, and community legal clinics cannot afford subscription AI. They serve the people who need help most, with the least resources.

Gemma 4 changes that equation. A self-hosted Jeremy can run on a single server, a donated laptop, or a legal aid nonprofit's existing infrastructure — anywhere in the world — at $0 per conversation, forever.

This is what open weights makes possible: **AI that serves people, not margins.**

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

Training ran on Kaggle free GPU (T4, 15.6GB VRAM, sm_75, fp16) in **5h 18m** at $0 cost. 927 steps across 3 epochs, effective batch size 16, gradient accumulation 16, ~25 sec/step, QLoRA r=16 α=32.
Final eval loss: 2.868 | Final training loss: 0.057

## Real-World Impact

Jeremy has already been used in active litigation:
- **McDaniel v. City of New York** (Bronx County Supreme Court, 2026) — Section 1983 civil rights case against 6 NYPD officers. Jeremy helped draft the complaint, demand letter, and preservation notices.
- Multiple pro se litigants guided through EEOC filings, contract disputes, and eviction defense across New York.

## What Gemma 4 Brings

Why Gemma 4 over other models?

1. **Open weights — the mission requirement.** No API. No per-token cost. No vendor lock-in. Any legal aid org in the world can self-host this.
2. **Size efficiency** — E4B runs on consumer hardware and mobile devices. Legal help shouldn't require a data center.
3. **128K context** — Full complaints, contracts, and case law fit in a single prompt.
4. **Instruction-tuned** — Gemma 4's instruction following maps directly to procedural legal guidance.
5. **Global deployability** — Fine-tune once for any jurisdiction, any language. The architecture is portable and self-hostable anywhere.

## Cost

| Component | Monthly Cost |
|-----------|-------------|
| Hosting (Oracle Free Tier) | $0 |
| Gemma 4 inference (self-hosted) | $0 |
| Training (Kaggle GPU) | $0 |
| Training data collection | $0 |
| Voice (Kokoro TTS, self-hosted) | $0 |
| **Total** | **$0** |

This is the number that matters for legal aid organizations. $0 to run. $0 per conversation. No budget approval needed.

## Try It

- **Live Demo:** [prosenetwork.org/demo](https://prosenetwork.org/demo)
- **Model:** [huggingface.co/israelburns/jeremy-gemma4](https://huggingface.co/israelburns/jeremy-gemma4)
- **HF Space Demo:** [huggingface.co/spaces/israelburns/jeremy-gemma4-demo](https://huggingface.co/spaces/israelburns/jeremy-gemma4-demo)
- **Code:** [github.com/israelburns/gemma4-good-hackathon](https://github.com/israelburns/gemma4-good-hackathon)

Ask Jeremy anything:
- "I was served with a lawsuit in New York. I have 20 days to respond. What do I do?"
- "My landlord is evicting me without proper notice in California."
- "How do I file an EEOC complaint for workplace discrimination?"

## Team

- **Israel "Ace" Burns** — Cornell Tech LLM '17, Co-Founder, New York
- **Esco Obong** — Co-Owner & Technical Lead, New York
