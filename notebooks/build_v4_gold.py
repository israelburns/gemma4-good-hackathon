#!/usr/bin/env python3
"""
build_v4_gold.py — Decomposed gold-dataset pipeline for Jeremy (Gemma-4B QLoRA).

Ace Burns Protocol build:
  Stage 2 (Kimi)   — per-source recipes, per-bucket dedup, anti-SCOTUS-dominance
  Stage 3 (Gemini) — caught that Kimi's recipes assumed LLM synthesis; hardened to
                     NATIVE-COLUMNS-ONLY, oversample-then-slice ordering
  Stage 4 (this)   — local $0 pipeline, no LLM calls

Target: EXACTLY 1,500 gold pairs.
  structured_to_guidance 450 | narrative_to_facts 450
  contract_to_clauses 375 | draft_document_section 225

Sources — ONLY those with a real >=150-char response column:
  v3_gold         native instruction/response pairs (the original gold)
  Lawyer-Instruct instruction/input -> instruction, output -> response
  CUAD            subtask name -> instruction, `text` (real clause) -> response
  LEDGAR          "draft a {label} clause" -> instruction, `text` -> response

SCOTUS / ECtHR / Reddit / learned_hands are DROPPED: they are classification
datasets with no native long-form response column — usable only via LLM
synthesis, which this $0 pipeline does not do (Gemini Stage 3 ruling).
"""

import os, re, json, hashlib, collections
from datasets import load_dataset
from huggingface_hub import hf_hub_download

HF_TOKEN  = os.environ.get("HF_TOKEN", "hf_REDACTED")
HF_REPO   = "israelburns/jeremy-training-data"
OUT_LOCAL = "/tmp/jeremy_training_v4_gold.jsonl"

BUCKETS = ["structured_to_guidance", "narrative_to_facts",
           "contract_to_clauses", "draft_document_section"]
BUCKET_TARGET = {
    "structured_to_guidance": 450,
    "narrative_to_facts":     450,
    "contract_to_clauses":    375,
    "draft_document_section": 225,
}

def log(m): print(f"  {m}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4.1 — OVERSAMPLED PULL (native columns only; ~3x target)
# ─────────────────────────────────────────────────────────────────────────────

DRAFT_KW = ("draft", "write a", "compose", "prepare a motion", "prepare an",
            "memorandum", "complaint", "petition", "clause", "provision")

def pull_v3_gold():
    p = hf_hub_download(HF_REPO, "jeremy_training_v3.jsonl",
                        repo_type="dataset", token=HF_TOKEN)
    rows = []
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        task = r.get("task", "")
        # safety_compliance is not one of our 4 buckets -> fold into structured
        bucket = task if task in BUCKETS else (
            "structured_to_guidance" if task == "safety_compliance" else None)
        if not bucket:
            continue
        rows.append({
            "task_type":    bucket,
            "instruction":  (r.get("instruction") or "").strip(),
            "input":        "",
            "response":     (r.get("response") or "").strip(),
            "source":       "v3_gold",
            "jurisdiction": r.get("jurisdiction", ""),
        })
    log(f"v3_gold: {len(rows)} rows")
    return rows

def pull_lawyer_instruct(cap=2500):
    ds = load_dataset("Alignment-Lab-AI/Lawyer-Instruct", split="train")
    rows = []
    for r in ds:
        ins = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        out = (r.get("output") or "").strip()
        if not ins or not out:
            continue
        is_draft = any(k in ins.lower() for k in DRAFT_KW)
        rows.append({
            "task_type":    "draft_document_section" if is_draft else "structured_to_guidance",
            "instruction":  ins if not inp else f"{ins}\n\nContext: {inp[:1500]}",
            "input":        "",
            "response":     out,
            "source":       "lawyer_instruct",
            "jurisdiction": "",
        })
        if len(rows) >= cap:
            break
    log(f"lawyer_instruct: {len(rows)} rows")
    return rows

# Human-readable instruction stems keyed by cuad subtask.
# Kimi Stage 5 fix A: the response is a RAW clause, not an explanation — so the
# instruction must say "extract", not "explain". Task aligned to response.
def _cuad_instruction(subtask):
    name = subtask.replace("cuad_", "").replace("_", " ").replace("-", "/")
    return (f"Extract the '{name}' clause from the following contract excerpt. "
            f"Return the exact clause text that governs this provision.")

CUAD_SUBTASKS = [
    "cuad_governing_law", "cuad_anti-assignment", "cuad_audit_rights",
    "cuad_cap_on_liability", "cuad_change_of_control", "cuad_competitive_restriction_exception",
    "cuad_covenant_not_to_sue", "cuad_effective_date", "cuad_exclusivity",
    "cuad_insurance", "cuad_ip_ownership_assignment", "cuad_irrevocable_or_perpetual_license",
    "cuad_license_grant", "cuad_liquidated_damages", "cuad_minimum_commitment",
    "cuad_most_favored_nation", "cuad_non-compete", "cuad_non-disparagement",
    "cuad_notice_period_to_terminate_renewal", "cuad_post-termination_services",
    "cuad_renewal_term", "cuad_revenue-profit_sharing", "cuad_rofr-rofo-rofn",
    "cuad_source_code_escrow", "cuad_termination_for_convenience",
    "cuad_uncapped_liability", "cuad_warranty_duration", "cuad_affiliate_license-licensee",
    "cuad_affiliate_license-licensor",
]

def pull_cuad():
    # Kimi Stage 5 fix A: CUAD test splits contain POSITIVE and NEGATIVE
    # excerpts (answer Yes/No). A negative excerpt does NOT contain the named
    # provision — training it as one teaches the model to hallucinate clauses.
    # Filter to answer == "Yes" only.
    rows, dropped_neg = [], 0
    for sub in CUAD_SUBTASKS:
        try:
            ds = load_dataset("nguha/legalbench", sub, split="test")
        except Exception as e:
            log(f"  cuad skip {sub}: {str(e)[:60]}")
            continue
        instr = _cuad_instruction(sub)
        for r in ds:
            if str(r.get("answer", "")).strip().lower() != "yes":
                dropped_neg += 1
                continue
            text = (r.get("text") or "").strip()
            if len(text) < 150:
                continue
            rows.append({
                "task_type":    "contract_to_clauses",
                "instruction":  instr,
                "input":        "",
                "response":     text,
                "source":       f"cuad:{sub}",
                "jurisdiction": "",
            })
    log(f"cuad: {len(rows)} rows (dropped {dropped_neg} negative-polarity excerpts)")
    return rows

def pull_ledgar(cap=2000):
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    names = ds.features["label"].names
    rows = []
    for r in ds:
        text = (r.get("text") or "").strip()
        label = r.get("label")
        if len(text) < 150 or label is None:
            continue
        lname = names[label] if isinstance(label, int) else str(label)
        rows.append({
            "task_type":    "draft_document_section",
            "instruction":  (f"Draft a '{lname}' clause suitable for inclusion "
                             f"in a formal legal agreement. The clause should be "
                             f"complete, precise, and ready to insert."),
            "input":        "",
            "response":     text,
            "source":       "ledgar",
            "jurisdiction": "",
        })
        if len(rows) >= cap:
            break
    log(f"ledgar: {len(rows)} rows")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4.1b — HALLUCINATION PURGE + LENGTH CAP  (Kimi Stage 5 fixes B + D)
# ─────────────────────────────────────────────────────────────────────────────

RESP_CAP = 2500   # chars — keeps responses inside a sane QLoRA token budget
_year_re = re.compile(r"\b(19|20)\d{2}\b")
_money_re = re.compile(r"\$[\d,]+")

def _hallucinates(instruction, response):
    """True if the response asserts a specific year or dollar figure that does
    NOT appear anywhere in the instruction — i.e. invented facts. Surgical: only
    flags hard, checkable tokens (years, money), not paraphrase."""
    instr_l = instruction.lower()
    for m in set(_year_re.findall(response)):
        # findall on the group returns the prefix only; re-scan for full years
        pass
    for yr in set(re.findall(r"\b(?:19|20)\d{2}\b", response)):
        if yr not in instruction:
            return True
    for amt in set(_money_re.findall(response)):
        if amt not in instruction:
            return True
    return False

def purge_and_cap(rows):
    kept, purged = [], 0
    for r in rows:
        # Hallucination purge — narrative_to_facts only (the bucket Kimi flagged:
        # v3_gold responses inventing dates/amounts not in the instruction).
        if r["task_type"] == "narrative_to_facts" and \
           _hallucinates(r["instruction"], r["response"]):
            purged += 1
            continue
        # Length cap — truncate over-long responses at a clean word boundary.
        if len(r["response"]) > RESP_CAP:
            cut = r["response"][:RESP_CAP].rsplit(" ", 1)[0]
            r["response"] = cut + " [...]"
        kept.append(r)
    log(f"hallucination purge: dropped {purged} narrative rows w/ invented years/amounts")
    log(f"length cap: responses truncated to <= {RESP_CAP} chars")
    return kept

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4.2 — QUALITY GATES
# ─────────────────────────────────────────────────────────────────────────────

LABEL_ONLY = {"yes", "no", "entailment", "contradiction", "true", "false",
              "neutral", "unrelated"}
VERB_SENT  = re.compile(r"[a-z]+\b[^.!?]*\b(is|are|was|were|be|has|have|had|"
                        r"shall|will|may|must|should|can|does|do|did|means|"
                        r"includes|applies|requires|provides|establishes|"
                        r"agrees|grants)\b", re.I)

# SEC EDGAR filing-dump markers — v3_gold contains raw SGML header junk in some
# contract rows (the exact garbage that polluted the prior build). Drop on sight.
_JUNK_MARKERS = ("ACCESSION NUMBER", ".hdr.sgml", "CONFORMED SUBMISSION TYPE",
                 "PUBLIC DOCUMENT COUNT", "CENTRAL INDEX KEY", "<SEC-DOCUMENT>",
                 "<SEC-HEADER>")

def _strip_entities(s):
    """Decode HTML entities (&#160; &amp; ...) and normalize preprocessing
    artifacts — CUAD masks elided clause spans with the literal token
    `<omitted>`; render it as a normal ellipsis so it reads as clause text."""
    import html
    s = html.unescape(s)
    s = s.replace("<omitted>", " ... ")
    return re.sub(r"\s{3,}", "  ", s)  # collapse runaway whitespace

def _is_junk(r):
    blob = r["instruction"] + " " + r["response"]
    if any(m in blob for m in _JUNK_MARKERS):
        return True
    # entity-soup: if HTML entities make up a real fraction of the text
    if blob.count("&#") > 8:
        return True
    return False

def passes_quality(r):
    ins, resp = r["instruction"], r["response"]
    if len(ins) < 60:                       return False
    if len(resp) < 150:                     return False
    if resp.strip().lower() in LABEL_ONLY:  return False
    if len(resp) < len(ins):                return False  # reversed/echo
    # near copy-paste: >80% of the shorter inside the longer
    a, b = sorted([ins.lower(), resp.lower()], key=len)
    if a and a[: int(len(a) * 0.8)] in b:   return False
    # Verb-sentence gate guards against label/heading responses — but
    # narrative_to_facts responses are structured JSON fact extractions BY
    # DESIGN, not prose. Applying it there wrongly nukes valid gold pairs.
    if r["task_type"] != "narrative_to_facts" and not VERB_SENT.search(resp):
        return False
    return True

def quality_filter(rows):
    # First strip HTML entities, then drop SGML/EDGAR junk, then quality gate.
    junk = 0
    cleaned = []
    for r in rows:
        r["instruction"] = _strip_entities(r["instruction"])
        r["response"]    = _strip_entities(r["response"])
        if _is_junk(r):
            junk += 1
            continue
        cleaned.append(r)
    log(f"junk filter: dropped {junk} SGML/entity-soup rows")
    kept = [r for r in cleaned if passes_quality(r)]
    log(f"quality gate: {len(cleaned)} -> {len(kept)} ({len(cleaned)-len(kept)} dropped)")
    return kept

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4.3 — EXACT (MD5) DEDUP
# ─────────────────────────────────────────────────────────────────────────────

def md5_dedup(rows):
    seen, kept = set(), []
    for r in rows:
        h = hashlib.md5((r["instruction"] + r["response"][:200]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    log(f"md5 dedup: {len(rows)} -> {len(kept)}")
    return kept

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4.4 — PER-BUCKET NEAR-DEDUP (word 5-gram Jaccard) + sort richest-first
#
# Pure-Python lexical near-dedup. The Python 3.14 env has a broken
# transformers/librosa/pkg_resources chain that blocks sentence-transformers;
# rather than fight it on a deadline, we use word-5-gram Jaccard, which is a
# strong fit for the actual dup profile here: the high-risk buckets
# (contract_to_clauses, draft_document_section) are templated legal boilerplate
# that shares heavy n-gram overlap when near-duplicate. Exact dupes are already
# gone (MD5). Threshold 0.55 ≈ aggressive (Kimi's fallback recommendation).
# ─────────────────────────────────────────────────────────────────────────────

JACCARD_THRESHOLD = 0.70   # 0.55 was too aggressive; 0.70 keeps near-dups apart
                           # without eating curated v3 prose
_word_re = re.compile(r"[a-z0-9]+")

# Per-bucket v3_gold ceiling — forces non-v3 sources into the mix for diversity
# (Kimi Stage 2). narrative is v3-only (no other source feeds it) so its ceiling
# is the full target. The others reserve room for lawyer_instruct / cuad / ledgar.
V3_CEILING = {
    "structured_to_guidance": 250,   # reserve ~200 for lawyer_instruct
    "narrative_to_facts":     450,   # v3-only bucket — no effective cap
    "contract_to_clauses":    175,   # reserve ~200 for cuad
    "draft_document_section":  90,   # reserve ~135 for ledgar + lawyer_instruct
}

def _ngrams(text, n=5):
    toks = _word_re.findall(text.lower())
    if len(toks) < n:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))

def _is_dup(g, kept_grams):
    for kg in kept_grams:
        if not g and not kg:
            return True
        inter = len(g & kg)
        if inter:
            union = len(g | kg)
            if union and inter / union >= JACCARD_THRESHOLD:
                return True
    return False

def semantic_dedup_and_slice(rows):
    final = []
    for bucket in BUCKETS:
        pool = [r for r in rows if r["task_type"] == bucket]
        pool.sort(key=lambda r: len(r["response"]), reverse=True)  # richest first
        if not pool:
            log(f"  {bucket}: EMPTY POOL"); continue
        target  = BUCKET_TARGET[bucket]
        ceiling = V3_CEILING[bucket]
        kept, kept_grams, v3_count = [], [], 0

        # PASS 1 — fill, respecting the v3_gold ceiling so non-v3 sources get in
        for r in pool:
            if len(kept) >= target:
                break
            is_v3 = r["source"] == "v3_gold"
            if is_v3 and v3_count >= ceiling:
                continue
            g = _ngrams(r["response"])
            if _is_dup(g, kept_grams):
                continue
            kept.append(r); kept_grams.append(g)
            if is_v3:
                v3_count += 1

        # PASS 2 — backfill from v3_gold above the ceiling if still short
        if len(kept) < target:
            for r in pool:
                if len(kept) >= target:
                    break
                if r["source"] != "v3_gold" or r in kept:
                    continue
                g = _ngrams(r["response"])
                if _is_dup(g, kept_grams):
                    continue
                kept.append(r); kept_grams.append(g)

        srcmix = dict(collections.Counter(r["source"].split(":")[0] for r in kept))
        log(f"  {bucket}: pool {len(pool)} -> kept {len(kept)}/{target} | {srcmix}")
        if len(kept) < target:
            log(f"  ⚠ {bucket} UNDERFILLED by {target - len(kept)}")
        final.extend(kept)
    return final

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== STAGE 4.1 — OVERSAMPLED PULL ===")
    raw = []
    raw += pull_v3_gold()
    raw += pull_lawyer_instruct()
    raw += pull_cuad()
    raw += pull_ledgar()
    print(f"  RAW POOL: {len(raw)}")
    print(f"  raw buckets: {dict(collections.Counter(r['task_type'] for r in raw))}")

    print("=== STAGE 4.1b — HALLUCINATION PURGE + LENGTH CAP ===")
    raw = purge_and_cap(raw)

    print("=== STAGE 4.2 — QUALITY GATES ===")
    rows = quality_filter(raw)

    print("=== STAGE 4.3 — MD5 DEDUP ===")
    rows = md5_dedup(rows)
    print(f"  post-md5 buckets: {dict(collections.Counter(r['task_type'] for r in rows))}")

    print("=== STAGE 4.4 — SEMANTIC DEDUP + REBALANCE ===")
    final = semantic_dedup_and_slice(rows)

    print("=== WRITE ===")
    with open(OUT_LOCAL, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist = dict(collections.Counter(r["task_type"] for r in final))
    srcs = dict(collections.Counter(r["source"].split(":")[0] for r in final))
    print(f"  TOTAL: {len(final)} pairs -> {OUT_LOCAL}")
    print(f"  buckets: {dist}")
    print(f"  sources: {srcs}")

if __name__ == "__main__":
    main()
