# ============================================================================
# build_v4_dataset.py
# Production V4 pipeline -> jeremy_training_v4.jsonl (24,000 gold pairs)
# Target: eval loss < 2.5, train loss > 0.5 (closes V13 overfit gap)
# Runtime: Google Colab, T4 GPU (15GB VRAM). No paid APIs. Heuristic-only QC.
# ============================================================================

# ---------------------------------------------------------------------------
# CELL 1 - Install
# ---------------------------------------------------------------------------
# !pip install -q "sentence-transformers>=3.1.0" datasketch huggingface_hub datasets pdfplumber

# ---------------------------------------------------------------------------
# CELL 2 - Imports, auth, config
# ---------------------------------------------------------------------------
import os
import re
import json
import hashlib
import random
import time
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, login, hf_hub_download
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH

try:
    from google.colab import userdata  # type: ignore
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as e:
        print(f"[WARN] HF login failed: {e}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_PATH       = "jeremy_training_v4.jsonl"
EVAL_OUTPUT_PATH  = "jeremy_eval_v4.jsonl"
SENTINEL_PATH     = "GATES_PASSED.v4"

HF_TARGET_REPO = "israelburns/jeremy-training-data"
HF_TARGET_FILE = "jeremy_training_v4.jsonl"
HF_EVAL_FILE   = "jeremy_eval_v4.jsonl"

EVAL_SPLIT_FRAC = 0.10
TOTAL_TARGET    = 24_000

TARGET_RATIOS = {
    "structured_to_guidance": 9_600,
    "narrative_to_facts":      6_000,
    "contract_to_clauses":     4_800,
    "draft_document_section":  3_600,
}

QUALITY_FLOORS = {
    "structured_to_guidance": 60,
    "narrative_to_facts":     55,
    "contract_to_clauses":    65,
    "draft_document_section": 45,
}

SOURCE_CAPS = {
    "reglab/housing_qa":                 ("structured_to_guidance", 6_000),
    "theatticusproject/acord":           ("contract_to_clauses",    2_500),
    "theatticusproject/cuad":            ("contract_to_clauses",    2_000),
    "kiddothe2b/contract-nli":           ("contract_to_clauses",    1_500),
    "Alignment-Lab-AI/Lawyer-Instruct":  ("structured_to_guidance", 6_000),
    "jonathanli/legal-advice-reddit":    ("narrative_to_facts",     6_000),
    "coastalcph/lex_glue":               ("narrative_to_facts",     5_000),
    "lex_glue_ledgar":                   ("contract_to_clauses",    5_000),
    "lex_glue_ledgar_draft":             ("draft_document_section", 4_000),
    "lex_glue_scotus":                   ("narrative_to_facts",     3_000),
    "lex_glue_unfair_tos":               ("contract_to_clauses",    3_000),
    "nguha/legalbench":                  ("structured_to_guidance", 8_000),
    "reglab/barexam_qa":                 ("structured_to_guidance", 1_500),
    "lawinstruct/lawinstruct":           ("narrative_to_facts",     1_000),
    "pile-of-law/pile-of-law":           ("structured_to_guidance", 1_500),
    "V3_synthetic":                      ("structured_to_guidance", 959),
    "V3_court_guides":                   ("draft_document_section", 111),
}

INTRA_TYPE_SEMANTIC_THRESHOLD = 0.93
INTER_TYPE_SEMANTIC_THRESHOLD = 0.95
TEMPLATE_JACCARD_THRESHOLD    = 0.85
LEAKAGE_COSINE_THRESHOLD      = 0.97
PARAPHRASE_LEAK_THRESHOLD     = 0.99
NGRAM_LEAK_SIZE               = 25

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def log(msg: str):
    print(msg, flush=True)


def gate_log(gate: str, name: str, passed: int, dropped: int, reason: str):
    log(f"[GATE {gate} - {name}] Passed: {passed} | Dropped: {dropped} | Reason: {reason}")


# ---------------------------------------------------------------------------
# CELL 3 - Source loaders
# Each loader returns List[Dict] with unified-ish schema; defensive on failure.
# ---------------------------------------------------------------------------

def _safe_load(name: str, *args, **kwargs):
    try:
        return load_dataset(name, *args, **kwargs)
    except Exception as e:
        log(f"[WARN] Failed to load {name} args={args} kwargs={kwargs}: {e}")
        return None


def load_housing_qa() -> List[Dict[str, Any]]:
    ds = _safe_load("reglab/housing_qa", split="train")
    for parquet_glob in [
        "hf://datasets/reglab/housing_qa/data/train-*.parquet",
        "hf://datasets/reglab/housing_qa/housing_qa/train-*.parquet",
        "hf://datasets/reglab/housing_qa/refs/convert/parquet/housing_qa/train/*.parquet",
        "hf://datasets/reglab/housing_qa/refs/convert/parquet/default/train/*.parquet",
    ]:
        if ds is not None:
            break
        ds = _safe_load("parquet", data_files={"train": parquet_glob}, split="train")
    if ds is None:
        log("[WARN] reglab/housing_qa unavailable — returning []")
        return []
    out = []
    for row in ds:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        statutes = row.get("statutes") or []
        cite = ""
        if statutes and isinstance(statutes, list):
            first = statutes[0]
            if isinstance(first, dict):
                cite = first.get("citation", "") or ""
        if not q or not a:
            continue
        resp = f"Answer: {a}"
        if cite:
            resp += f" Statute: {cite}"
        out.append({
            "task_type":    "structured_to_guidance",
            "instruction":  q,
            "input":        "",
            "response":     resp,
            "source":       "reglab/housing_qa",
            "jurisdiction": row.get("state", "") or "",
        })
    return out


def load_acord() -> List[Dict[str, Any]]:
    ds = _safe_load("theatticusproject/acord", split="train")
    for parquet_glob in [
        "hf://datasets/theatticusproject/acord/data/train-*.parquet",
        "hf://datasets/theatticusproject/acord/refs/convert/parquet/default/train/*.parquet",
        "hf://datasets/theatticusproject/acord/refs/convert/parquet/acord/train/*.parquet",
    ]:
        if ds is not None:
            break
        ds = _safe_load("parquet", data_files={"train": parquet_glob}, split="train")
    if ds is None:
        ds = _safe_load("theatticusproject/acord", split="test")
    if ds is None:
        log("[WARN] theatticusproject/acord unavailable — returning []")
        return []
    out = []
    for row in ds:
        rel = row.get("relevance_score", 0) or 0
        try:
            rel = int(rel)
        except Exception:
            rel = 0
        if rel < 4:
            continue
        query  = (row.get("query") or "").strip()
        clause = (row.get("clause") or "").strip()
        if not query or not clause:
            continue
        out.append({
            "task_type":    "contract_to_clauses",
            "instruction":  f"Find a clause that: {query}",
            "input":        "",
            "response":     clause,
            "source":       "theatticusproject/acord",
            "jurisdiction": "",
        })
    return out


def load_cuad() -> List[Dict[str, Any]]:
    # Use no_checks to bypass split-size verification (PDF extraction yields partial rows).
    # Drop the 'document' (PDF) column before iterating — avoids pdfplumber ImportError.
    ds = _safe_load("theatticusproject/cuad", split="train", verification_mode="no_checks")
    for parquet_glob in [
        "hf://datasets/theatticusproject/cuad/data/train-*.parquet",
        "hf://datasets/theatticusproject/cuad/refs/convert/parquet/default/train/*.parquet",
        "hf://datasets/theatticusproject/cuad/refs/convert/parquet/cuad/train/*.parquet",
    ]:
        if ds is not None:
            break
        ds = _safe_load("parquet", data_files={"train": parquet_glob}, split="train")
    if ds is None:
        log("[WARN] theatticusproject/cuad unavailable — returning []")
        return []
    # Remove PDF feature column so iteration doesn't require pdfplumber
    pdf_cols = [c for c in ds.column_names if c in ("document", "pdf", "file")]
    if pdf_cols:
        ds = ds.remove_columns(pdf_cols)
    out = []
    for row in ds:
        q   = (row.get("question") or "").strip()
        ctx = (row.get("context") or "").strip()
        answers = row.get("answers") or {}
        texts = answers.get("text") if isinstance(answers, dict) else None
        if not texts:
            continue
        resp = (texts[0] or "").strip()
        if not q or not resp:
            continue
        out.append({
            "task_type":    "contract_to_clauses",
            "instruction":  q,
            "input":        ctx[:4000],
            "response":     resp,
            "source":       "theatticusproject/cuad",
            "jurisdiction": "",
        })
    return out


def load_contract_nli() -> List[Dict[str, Any]]:
    ds = _safe_load("kiddothe2b/contract-nli", split="train")
    for parquet_glob in [
        "hf://datasets/kiddothe2b/contract-nli/data/train-*.parquet",
        "hf://datasets/kiddothe2b/contract-nli/contract-nli/train-*.parquet",
        "hf://datasets/kiddothe2b/contract-nli/refs/convert/parquet/default/train/*.parquet",
        "hf://datasets/kiddothe2b/contract-nli/refs/convert/parquet/contract-nli/train/*.parquet",
    ]:
        if ds is not None:
            break
        ds = _safe_load("parquet", data_files={"train": parquet_glob}, split="train")
    if ds is None:
        log("[WARN] kiddothe2b/contract-nli unavailable — returning []")
        return []
    out = []
    for row in ds:
        hyp   = (row.get("hypothesis") or "").strip()
        prem  = (row.get("premise") or "").strip()
        label = row.get("label", "")
        ev    = row.get("evidence_spans") or []
        ev_text = ""
        if ev and isinstance(ev, list):
            first = ev[0]
            if isinstance(first, dict):
                ev_text = first.get("text", "") or ""
            elif isinstance(first, str):
                ev_text = first
        if not hyp or not prem:
            continue
        resp = f"{label}"
        if ev_text:
            resp += f" Evidence: {ev_text}"
        out.append({
            "task_type":    "contract_to_clauses",
            "instruction":  f"Does this contract contain a clause that {hyp}?",
            "input":        prem[:4000],
            "response":     resp,
            "source":       "kiddothe2b/contract-nli",
            "jurisdiction": "",
        })
    return out


def load_lawyer_instruct() -> List[Dict[str, Any]]:
    ds = _safe_load("Alignment-Lab-AI/Lawyer-Instruct", split="train")
    if ds is None:
        return []
    out = []
    drafting_kw = ("draft", "write a", "compose", "prepare a motion",
                   "prepare an", "memorandum", "complaint", "petition")
    for row in ds:
        ins  = (row.get("instruction") or "").strip()
        inp  = (row.get("input") or "").strip()
        outp = (row.get("output") or "").strip()
        if not ins or not outp:
            continue
        is_draft = any(k in ins.lower() for k in drafting_kw)
        ttype = "draft_document_section" if is_draft else "structured_to_guidance"
        out.append({
            "task_type":    ttype,
            "instruction":  ins,
            "input":        inp[:4000],
            "response":     outp,
            "source":       "Alignment-Lab-AI/Lawyer-Instruct",
            "jurisdiction": "",
        })
    return out


def load_legal_advice_reddit() -> List[Dict[str, Any]]:
    ds = _safe_load("jonathanli/legal-advice-reddit", split="train")
    if ds is None:
        return []
    out = []
    for row in ds:
        title = (row.get("title") or "").strip()
        body  = (row.get("body") or "").strip()
        flair = str(row.get("flair_label") or "")
        if not title or len(body) < 100:
            continue
        out.append({
            "task_type":    "narrative_to_facts",
            "instruction":  "Extract the legal issue, jurisdiction signals, and key facts from this post.",
            "input":        f"{title}\n{body}"[:4000],
            "response":     f"Legal issue: {flair}. Key facts: {body[:500]}",
            "source":       "jonathanli/legal-advice-reddit",
            "jurisdiction": "",
        })
    return out


def load_lex_glue_ecthr() -> List[Dict[str, Any]]:
    ds = _safe_load("coastalcph/lex_glue", "ecthr_a", split="train")
    if ds is None:
        return []
    out = []
    for row in ds:
        texts = row.get("text") or []
        if isinstance(texts, list):
            joined = " ".join([t for t in texts if isinstance(t, str)])
        else:
            joined = str(texts)
        labels = row.get("labels") or row.get("label") or []
        if not joined.strip():
            continue
        out.append({
            "task_type":    "narrative_to_facts",
            "instruction":  "List the facts alleged and violations claimed.",
            "input":        joined[:4000],
            "response":     f"Alleged violations: {labels}",
            "source":       "coastalcph/lex_glue",
            "jurisdiction": "ECHR",
        })
    return out


# ledgar: 60K contractual provisions with clause-type labels.
# Labels map to two task types: clause extraction (contract_to_clauses) and
# section drafting (draft_document_section).
_LEDGAR_DRAFT_LABELS = frozenset([
    "Governing Laws", "Definitions", "Miscellaneous", "Preambles",
    "Remedies", "Notices", "Amendments", "Waivers", "Entire Agreements",
    "Severabilities", "Counterparts", "Headings", "Further Assurances",
    "Representations", "Disclosures",
])

def _load_ledgar_rows() -> List[Dict[str, Any]]:
    out_contract, out_draft = [], []
    for split_name in ("train", "validation", "test"):
        ds = _safe_load("coastalcph/lex_glue", "ledgar", split=split_name)
        if ds is None:
            continue
        for row in ds:
            text = (row.get("text") or "").strip()
            label = str(row.get("label") or "").strip()
            if not text or len(text) < 50 or not label:
                continue
            if label in _LEDGAR_DRAFT_LABELS:
                out_draft.append({
                    "task_type":    "draft_document_section",
                    "instruction":  f"Draft a '{label}' section for a legal contract.",
                    "input":        "",
                    "response":     text[:2000],
                    "source":       "lex_glue_ledgar_draft",
                    "jurisdiction": "US",
                })
            else:
                out_contract.append({
                    "task_type":    "contract_to_clauses",
                    "instruction":  f"Extract and explain the '{label}' clause in this contract.",
                    "input":        "",
                    "response":     text[:2000],
                    "source":       "lex_glue_ledgar",
                    "jurisdiction": "US",
                })
    return out_contract, out_draft

def load_lex_glue_ledgar() -> List[Dict[str, Any]]:
    rows, _ = _load_ledgar_rows()
    return rows

def load_lex_glue_ledgar_draft() -> List[Dict[str, Any]]:
    _, rows = _load_ledgar_rows()
    return rows

def load_lex_glue_scotus() -> List[Dict[str, Any]]:
    out = []
    for split_name in ("train", "validation", "test"):
        ds = _safe_load("coastalcph/lex_glue", "scotus", split=split_name)
        if ds is None:
            continue
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text or len(text) < 100:
                continue
            out.append({
                "task_type":    "narrative_to_facts",
                "instruction":  "Summarize the key legal facts and the court's holding in this Supreme Court opinion.",
                "input":        "",
                "response":     text[:2000],
                "source":       "lex_glue_scotus",
                "jurisdiction": "US",
            })
    return out

def load_lex_glue_unfair_tos() -> List[Dict[str, Any]]:
    out = []
    for split_name in ("train", "validation", "test"):
        ds = _safe_load("coastalcph/lex_glue", "unfair_tos", split=split_name)
        if ds is None:
            continue
        for row in ds:
            text = (row.get("text") or "").strip()
            label = str(row.get("label") or "").strip()
            if not text or len(text) < 30:
                continue
            inst = (
                f"Identify the potentially unfair '{label}' clause in this Terms of Service excerpt and explain the consumer risk."
                if label and label not in ("O", "0", "")
                else "Identify any potentially unfair clause in this Terms of Service excerpt and explain the consumer risk."
            )
            out.append({
                "task_type":    "contract_to_clauses",
                "instruction":  inst,
                "input":        "",
                "response":     text[:2000],
                "source":       "lex_glue_unfair_tos",
                "jurisdiction": "US",
            })
    return out


_LB_CONTRACT_SUBS = frozenset([
    "cuad_affiliate_license-licensee", "cuad_affiliate_license-licensor",
    "cuad_anti-assignment", "cuad_audit_rights", "cuad_cap_on_liability",
    "cuad_change_of_control", "cuad_competitive_restriction_exception",
    "cuad_covenant_not_to_sue", "cuad_effective_date", "cuad_exclusivity",
    "cuad_expiration_date", "cuad_governing_law", "cuad_insurance",
    "cuad_ip_ownership_assignment", "cuad_irrevocable_or_perpetual_license",
    "cuad_joint_ip_ownership", "cuad_license_grant", "cuad_liquidated_damages",
    "cuad_minimum_commitment", "cuad_most_favored_nation",
    "cuad_no-solicit_of_customers", "cuad_no-solicit_of_employees",
    "cuad_non-compete", "cuad_non-disparagement",
    "cuad_non-transferable_license", "cuad_notice_period_to_terminate_renewal",
    "cuad_post-termination_services", "cuad_price_restrictions",
    "cuad_renewal_term", "cuad_revenue-profit_sharing", "cuad_rofr-rofo-rofn",
    "cuad_source_code_escrow", "cuad_termination_for_convenience",
    "cuad_third_party_beneficiary", "cuad_uncapped_liability",
    "cuad_unlimited-all-you-can-eat-license", "cuad_volume_restriction",
    "cuad_warranty_duration",
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_inclusion_of_verbally_conveyed_information",
    "contract_nli_limited_use", "contract_nli_no_licensing",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_permissible_acquirement_of_similar_information",
    "contract_nli_permissible_copy",
    "contract_nli_permissible_development_of_similar_information",
    "contract_nli_permissible_post-agreement_possession",
    "contract_nli_return_of_confidential_information",
    "contract_nli_sharing_with_employees",
    "contract_nli_sharing_with_third-parties",
    "contract_nli_survival_of_obligations",
    "contract_qa", "consumer_contracts_qa",
])
_LB_NARRATIVE_SUBS = frozenset([
    "learned_hands_housing", "learned_hands_employment",
    "learned_hands_courts", "learned_hands_divorce",
    "learned_hands_consumer", "learned_hands_benefits",
    "learned_hands_crime", "learned_hands_health",
    "learned_hands_immigration", "learned_hands_torts",
    "learned_hands_family", "learned_hands_domestic_violence",
    "learned_hands_traffic", "learned_hands_estates",
    "learned_hands_education", "learned_hands_business",
])


def load_legalbench() -> List[Dict[str, Any]]:
    subtasks = list(_LB_NARRATIVE_SUBS) + [
        "rule_qa", "hearsay", "overruling", "personal_jurisdiction",
    ]
    all_subs = subtasks + list(_LB_CONTRACT_SUBS)
    out = []
    for sub in all_subs:
        # Collect both train and test — legalbench train splits are tiny (4-8 rows);
        # test splits hold the bulk (50-2400 rows). Combine both.
        rows_for_sub = []
        for split_name in ("train", "test"):
            ds = _safe_load("nguha/legalbench", sub, split=split_name)
            if ds is not None:
                rows_for_sub.extend(list(ds))
        if not rows_for_sub:
            continue
        if sub in _LB_CONTRACT_SUBS:
            ttype = "contract_to_clauses"
        elif sub in _LB_NARRATIVE_SUBS:
            ttype = "narrative_to_facts"
        else:
            ttype = "structured_to_guidance"
        for row in rows_for_sub:
            if ttype == "contract_to_clauses":
                # cuad_* and contract_nli_* rows: answer is Yes/No or Entailment (too short).
                # Use the 'text' field — the actual contract clause excerpt — as the response.
                clause_text = (row.get("text") or "").strip()
                if len(clause_text) < 50:
                    continue
                clause_topic = (sub.replace("cuad_", "")
                                   .replace("contract_nli_", "")
                                   .replace("_", " ").replace("-", " "))
                inst = f"Extract and identify the clause about '{clause_topic}' in the following contract text."
                out.append({
                    "task_type":    "contract_to_clauses",
                    "instruction":  inst,
                    "input":        "",
                    "response":     clause_text[:2000],
                    "source":       f"nguha/legalbench:{sub}",
                    "jurisdiction": "US",
                })
            else:
                q = (row.get("question") or row.get("text") or row.get("input") or "").strip()
                a = (row.get("answer") or row.get("label") or row.get("output") or "").strip()
                if not q or not a:
                    continue
                out.append({
                    "task_type":    ttype,
                    "instruction":  q[:2000],
                    "input":        "",
                    "response":     a[:2000],
                    "source":       f"nguha/legalbench:{sub}",
                    "jurisdiction": "US",
                })
    return out


def load_barexam_qa() -> List[Dict[str, Any]]:
    ds = _safe_load("reglab/barexam_qa", split="train")
    for parquet_glob in [
        "hf://datasets/reglab/barexam_qa/data/train-*.parquet",
        "hf://datasets/reglab/barexam_qa/barexam_qa/train-*.parquet",
        "hf://datasets/reglab/barexam_qa/refs/convert/parquet/default/train/*.parquet",
    ]:
        if ds is not None:
            break
        ds = _safe_load("parquet", data_files={"train": parquet_glob}, split="train")
    if ds is None:
        return []
    out = []
    for row in ds:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        out.append({
            "task_type":    "structured_to_guidance",
            "instruction":  q,
            "input":        "",
            "response":     a,
            "source":       "reglab/barexam_qa",
            "jurisdiction": "US",
        })
    return out


def load_lawinstruct() -> List[Dict[str, Any]]:
    ds = _safe_load("lawinstruct/lawinstruct", split="train")
    for parquet_glob in [
        "hf://datasets/lawinstruct/lawinstruct/data/train-*.parquet",
        "hf://datasets/lawinstruct/lawinstruct/refs/convert/parquet/default/train/*.parquet",
    ]:
        if ds is not None:
            break
        ds = _safe_load("parquet", data_files={"train": parquet_glob}, split="train")
    if ds is None:
        return []
    out = []
    for row in ds:
        lang  = (row.get("language") or "").lower()
        juris = (row.get("jurisdiction") or "").lower()
        if lang != "en":
            continue
        juris_norm = (juris or "").lower().replace(".", "").replace(" ", "")
        if "us" not in juris_norm and "unitedstates" not in juris_norm:
            continue
        q = (row.get("question") or row.get("instruction") or "").strip()
        a = (row.get("answer") or row.get("output") or "").strip()
        if not q or not a:
            continue
        out.append({
            "task_type":    "narrative_to_facts",
            "instruction":  q[:2000],
            "input":        "",
            "response":     a[:4000],
            "source":       "lawinstruct/lawinstruct",
            "jurisdiction": juris,
        })
    return out


def load_pile_of_law_legaladvice() -> List[Dict[str, Any]]:
    ds = _safe_load("pile-of-law/pile-of-law", "r_legaladvice", split="train")
    for parquet_glob in [
        "hf://datasets/pile-of-law/pile-of-law/r_legaladvice/train-*.parquet",
        "hf://datasets/pile-of-law/pile-of-law/data/r_legaladvice/train-*.parquet",
        "hf://datasets/pile-of-law/pile-of-law/refs/convert/parquet/r_legaladvice/train/*.parquet",
    ]:
        if ds is not None:
            break
        ds = _safe_load("parquet", data_files={"train": parquet_glob}, split="train")
    if ds is None:
        return []
    out = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        parts = text.split("\n\n", 2)
        if len(parts) < 2:
            continue
        instr = parts[0].strip()[:1000]
        resp  = "\n\n".join(parts[1:]).strip()[:3000]
        if len(instr) < 20 or len(resp) < 40:
            continue
        out.append({
            "task_type":    "structured_to_guidance",
            "instruction":  instr,
            "input":        "",
            "response":     resp,
            "source":       "pile-of-law/pile-of-law:r_legaladvice",
            "jurisdiction": "US",
        })
    return out


def load_v3_gold() -> List[Dict[str, Any]]:
    """V3 synthetic_rules + court_self_help passes through directly."""
    out = []
    try:
        path = hf_hub_download(
            repo_id="israelburns/jeremy-training-data",
            filename="jeremy_training_v3.jsonl",
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        log(f"[WARN] V3 gold load failed: {e}")
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                src = row.get("source", "")
                ttype = row.get("task_type") or "structured_to_guidance"
                # Accept all V3 rows; tag by source name or task_type
                if src == "synthetic_rules" or (src not in ("court_self_help",) and ttype != "draft_document_section"):
                    tagged_src = "V3_synthetic"
                else:
                    tagged_src = "V3_court_guides"
                out.append({
                    "task_type":    ttype,
                    "instruction":  row.get("instruction", ""),
                    "input":        row.get("input", ""),
                    "response":     row.get("response", row.get("output", "")),
                    "source":       tagged_src,
                    "jurisdiction": row.get("jurisdiction", ""),
                })
    except Exception as e:
        log(f"[WARN] V3 gold parse failed: {e}")
    return out


ALL_LOADERS = [
    ("reglab/housing_qa",                load_housing_qa),
    ("theatticusproject/acord",          load_acord),
    ("theatticusproject/cuad",           load_cuad),
    ("kiddothe2b/contract-nli",          load_contract_nli),
    ("Alignment-Lab-AI/Lawyer-Instruct", load_lawyer_instruct),
    ("jonathanli/legal-advice-reddit",   load_legal_advice_reddit),
    ("coastalcph/lex_glue",              load_lex_glue_ecthr),
    ("lex_glue_ledgar",                  load_lex_glue_ledgar),
    ("lex_glue_ledgar_draft",            load_lex_glue_ledgar_draft),
    ("lex_glue_scotus",                  load_lex_glue_scotus),
    ("lex_glue_unfair_tos",              load_lex_glue_unfair_tos),
    ("nguha/legalbench",                 load_legalbench),
    ("reglab/barexam_qa",                load_barexam_qa),
    ("lawinstruct/lawinstruct",          load_lawinstruct),
    ("pile-of-law/pile-of-law",          load_pile_of_law_legaladvice),
    ("V3_gold",                          load_v3_gold),
]


# ---------------------------------------------------------------------------
# CELL 4 - Schema normalize + per-source cap
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["task_type", "instruction", "response", "source"]


def normalize_and_cap() -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    log("[LOAD] Pulling all sources...")
    for name, loader in ALL_LOADERS:
        t0 = time.time()
        rows = loader()
        elapsed = time.time() - t0
        log(f"  [LOAD] {name}: {len(rows)} rows ({elapsed:.1f}s)")
        cap_info = SOURCE_CAPS.get(name)
        if cap_info is not None:
            _, cap = cap_info
            if len(rows) == 0 and cap >= 500:
                log(f"[WARN] Source {name} returned 0 rows — check schema or dataset availability")
            if len(rows) > cap:
                random.Random(SEED).shuffle(rows)
                rows = rows[:cap]
                log(f"    capped to {cap}")
        for r in rows:
            norm = {
                "task_type":     (r.get("task_type") or "").strip(),
                "instruction":   (r.get("instruction") or "").strip(),
                "input":         (r.get("input") or "").strip(),
                "response":      (r.get("response") or "").strip(),
                "source":        (r.get("source") or name).strip(),
                "jurisdiction":  (r.get("jurisdiction") or "").strip(),
                "quality_score": 0,
            }
            all_rows.append(norm)
    log(f"[LOAD] Total normalized rows: {len(all_rows)}")
    # Second pass: cap by row source tag (handles V3_gold split sources)
    for source_key, (expected_type, cap) in SOURCE_CAPS.items():
        source_rows = [r for r in all_rows if r.get("source") == source_key]
        if len(source_rows) > cap:
            random.Random(SEED).shuffle(source_rows)
            source_rows = source_rows[:cap]
            all_rows = [r for r in all_rows if r.get("source") != source_key] + source_rows
    return all_rows


# ---------------------------------------------------------------------------
# CELL 5 - Eval split (BEFORE any dedup or filtering)
# Stratified 90/10 by task_type, seed=42
# ---------------------------------------------------------------------------

def stratified_eval_split(rows: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["task_type"]].append(r)
    train, evalset = [], []
    rng = random.Random(SEED)
    for ttype, items in by_type.items():
        idx = list(range(len(items)))
        rng.shuffle(idx)
        n_eval = max(1, int(len(items) * EVAL_SPLIT_FRAC))
        eval_idx = set(idx[:n_eval])
        for i, it in enumerate(items):
            (evalset if i in eval_idx else train).append(it)
    log(f"[SPLIT] Train pool: {len(train)} | Eval pool: {len(evalset)}")
    return train, evalset


# ---------------------------------------------------------------------------
# CELL 6 - Gate G0: schema validation
# ---------------------------------------------------------------------------

def gate_g0_schema(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n_in = len(rows)
    kept = []
    dropped_reasons: Counter = Counter()
    for r in rows:
        missing = [f for f in REQUIRED_FIELDS if not r.get(f)]
        if missing:
            dropped_reasons[f"missing:{','.join(missing)}"] += 1
            continue
        if r["task_type"] not in TARGET_RATIOS:
            dropped_reasons["bad_task_type"] += 1
            continue
        if len(r["instruction"]) < 10 or len(r["response"]) < 10:
            dropped_reasons["too_short"] += 1
            continue
        kept.append(r)
    dropped = n_in - len(kept)
    pct_dropped = dropped / max(1, n_in)
    reason = f"top_drops={dict(dropped_reasons.most_common(3))}"
    gate_log("G0", "SCHEMA", len(kept), dropped, reason)
    if pct_dropped > 0.05:
        log(f"[GATE G0] WARNING: dropped {pct_dropped:.1%} (>5% threshold).")
    return kept


# ---------------------------------------------------------------------------
# CELL 7 - Gate G1: exact hash dedup
# ---------------------------------------------------------------------------

def _row_hash(r: Dict[str, Any]) -> str:
    key = (r["instruction"].strip().lower()
           + "||" + r["input"][:200].strip().lower()
           + "||" + r["response"][:200].strip().lower())
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def gate_g1_exact(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    kept = []
    for r in rows:
        h = _row_hash(r)
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    dropped = len(rows) - len(kept)
    gate_log("G1", "EXACT_HASH", len(kept), dropped,
             "md5(instruction+input[:200]+response[:200])")
    return kept


# ---------------------------------------------------------------------------
# CELL 8 - Gate G2: quality scoring (heuristic, 0-100, $0 cost)
# ---------------------------------------------------------------------------

_CITATION_RX = re.compile(
    r"(\b\d+\s+U\.?S\.?C\.?\s*§?\s*\d+|"
    r"\b§+\s*\d+(\.\d+)*|"
    r"\b\d+\s+C\.?F\.?R\.?\s*§?\s*\d+|"
    r"\bFed\.?\s*R\.?\s*(Civ|Crim|App|Evid)\.?\s*P\.?\s*\d+|"
    r"\b\d+\s+[A-Z][a-z]+\.?\s+\d+|"
    r"\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+)"
)
_MODAL_RX        = re.compile(r"\b(shall|must|may|will|should|cannot|may not|shall not)\b", re.I)
_STRUCT_RX       = re.compile(r"(^\s*\d+[\.\)]|^\s*[-•*]|\bStep \d+|\bFirst,|\bSecond,|\bThird,)", re.M)
_DEFINED_TERM_RX = re.compile(r'"[A-Z][A-Za-z ]+"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b\s*\(')
_HEADER_RX       = re.compile(r"^\s*(#+\s|[A-Z][A-Z \-]{4,}$|\d+\.\s+[A-Z])", re.M)
_DATE_RX         = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}")
_FLUFF_RX        = re.compile(r"\b(basically|essentially|just|simply|in conclusion|as you can see|in summary)\b", re.I)
_FACT_OPINION_RX = re.compile(r"\b(I think|I believe|in my opinion|probably|maybe|seems like|imo|imho)\b", re.I)
_PROC_RX         = re.compile(r"\b(file|petition|motion|hearing|service|served|complaint|answer|discovery|subpoena|affidavit|notarize|exhibit|brief|order|judgment)\b", re.I)


def _score_structured_to_guidance(r: Dict[str, Any]) -> int:
    text = r["response"]
    if not text:
        return 0
    score = 50
    actions = len(_PROC_RX.findall(text))
    sents = max(1, text.count(".") + text.count("\n"))
    action_density = actions / sents
    score += min(20, int(action_density * 40))
    structs = len(_STRUCT_RX.findall(text))
    score += min(15, structs * 3)
    if r.get("jurisdiction"):
        score += 10
    if _CITATION_RX.search(text):
        score += 10
    score -= len(_FLUFF_RX.findall(text)) * 4
    L = len(text)
    if L < 80:
        score -= 20
    elif L > 4000:
        score -= 10
    return max(0, min(100, score))


def _score_narrative_to_facts(r: Dict[str, Any]) -> int:
    text = r["response"] + " " + r["input"]
    if not text.strip():
        return 0
    score = 50
    caps = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text))
    words = max(1, len(text.split()))
    score += min(20, int((caps / words) * 200))
    score += min(15, len(_DATE_RX.findall(text)) * 4)
    score -= len(_FACT_OPINION_RX.findall(text)) * 5
    if _CITATION_RX.search(text):
        score += 8
    if len(r["response"]) < 60:
        score -= 20
    return max(0, min(100, score))


def _score_contract_to_clauses(r: Dict[str, Any]) -> int:
    text = r["response"]
    if not text:
        return 0
    score = 55
    score += min(20, len(_MODAL_RX.findall(text)) * 3)
    score += min(15, len(_DEFINED_TERM_RX.findall(text)) * 2)
    refs = len(re.findall(r"\b(Section|Clause|Article|Paragraph)\s+\d", text))
    score += min(10, refs * 3)
    score -= len(_FLUFF_RX.findall(text)) * 5
    if len(text) < 40:
        score -= 25
    return max(0, min(100, score))


def _score_draft_document_section(r: Dict[str, Any]) -> int:
    text = r["response"]
    if not text:
        return 0
    score = 50
    score += min(20, len(_HEADER_RX.findall(text)) * 5)
    score += min(15, len(_CITATION_RX.findall(text)) * 5)
    score += min(15, len(_PROC_RX.findall(text)) * 2)
    L = len(text)
    if L < 200:
        score -= 20
    elif L > 6000:
        score -= 5
    score -= len(_FLUFF_RX.findall(text)) * 4
    return max(0, min(100, score))


_SCORERS = {
    "structured_to_guidance": _score_structured_to_guidance,
    "narrative_to_facts":     _score_narrative_to_facts,
    "contract_to_clauses":    _score_contract_to_clauses,
    "draft_document_section": _score_draft_document_section,
}


def gate_g2_quality(rows: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Returns (passed, warm_reserve, cold_dropped)."""
    passed, warm, cold = [], [], []
    for r in rows:
        scorer = _SCORERS.get(r["task_type"])
        if scorer is None:
            cold.append(r)
            continue
        s = scorer(r)
        r["quality_score"] = s
        floor = QUALITY_FLOORS[r["task_type"]]
        if s >= floor:
            passed.append(r)
        elif s >= floor - 10:
            warm.append(r)
        else:
            cold.append(r)
    gate_log("G2", "QUALITY", len(passed), len(warm) + len(cold),
             f"floors={QUALITY_FLOORS} | warm={len(warm)} | cold={len(cold)}")
    return passed, warm, cold


# ---------------------------------------------------------------------------
# CELL 9 - Gate G3: sort by quality DESC, then 3-stage dedup kill chain
# ---------------------------------------------------------------------------

def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _semantic_dedup(rows: List[Dict[str, Any]],
                    model: SentenceTransformer,
                    threshold: float) -> List[Dict[str, Any]]:
    if len(rows) <= 1:
        return rows
    texts = [(r["instruction"] + " " + r["response"])[:1500] for r in rows]
    emb = _embed_texts(model, texts)
    n = len(rows)
    kept_mask = np.ones(n, dtype=bool)
    CHUNK = 512
    for i in range(0, n, CHUNK):
        if not kept_mask[i:i + CHUNK].any():
            continue
        block = emb[i:i + CHUNK]
        sims = block @ emb.T
        for local_i in range(block.shape[0]):
            global_i = i + local_i
            if not kept_mask[global_i]:
                continue
            row_sims = sims[local_i]
            dup_idx = np.where(row_sims >= threshold)[0]
            for j in dup_idx:
                if j > global_i and kept_mask[j]:
                    kept_mask[j] = False
    return [r for r, k in zip(rows, kept_mask) if k]


def _minhash_for(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) >= 5:
        for i in range(len(tokens) - 4):
            shingle = " ".join(tokens[i:i + 5])
            m.update(shingle.encode("utf-8"))
    elif tokens:
        m.update(" ".join(tokens).encode("utf-8"))
    return m


def _template_spam_dedup(rows: List[Dict[str, Any]], jaccard_threshold: float) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    num_perm = 64
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    kept = []
    for idx, r in enumerate(rows):
        key = f"r{idx}"
        m = _minhash_for(r["response"][:2000], num_perm=num_perm)
        if lsh.query(m):
            continue
        lsh.insert(key, m)
        kept.append(r)
    return kept


def gate_g3_semantic(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    log("[GATE G3] Loading MiniLM...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Sort DESC by quality so best exemplar per cluster survives
    rows_sorted = sorted(rows, key=lambda r: (r["quality_score"], len(r.get("instruction", ""))), reverse=True)
    n_in = len(rows_sorted)

    # Stage A: intra-type semantic dedup
    by_type = defaultdict(list)
    for r in rows_sorted:
        by_type[r["task_type"]].append(r)
    after_intra = []
    for ttype, items in by_type.items():
        kept = _semantic_dedup(items, model, INTRA_TYPE_SEMANTIC_THRESHOLD)
        log(f"  [G3a intra-{ttype}] {len(items)} -> {len(kept)}")
        after_intra.extend(kept)

    # Stage B: inter-type at higher threshold
    after_intra_sorted = sorted(after_intra, key=lambda r: (r["quality_score"], len(r.get("instruction", ""))), reverse=True)
    after_inter = _semantic_dedup(after_intra_sorted, model, INTER_TYPE_SEMANTIC_THRESHOLD)
    log(f"  [G3b inter-type] {len(after_intra_sorted)} -> {len(after_inter)}")

    # Stage C: template spam via MinHashLSH on response
    after_template = _template_spam_dedup(after_inter, TEMPLATE_JACCARD_THRESHOLD)
    log(f"  [G3c template-spam] {len(after_inter)} -> {len(after_template)}")

    gate_log("G3", "SEMANTIC_DEDUP",
             len(after_template), n_in - len(after_template),
             f"intra={INTRA_TYPE_SEMANTIC_THRESHOLD} inter={INTER_TYPE_SEMANTIC_THRESHOLD} template={TEMPLATE_JACCARD_THRESHOLD}")
    return after_template


# ---------------------------------------------------------------------------
# CELL 10 - Gate G4: rebalance to 24k targets, fill from warm reserve
# ---------------------------------------------------------------------------

def gate_g4_rebalance(passed: List[Dict[str, Any]],
                      warm_reserve: List[Dict[str, Any]]
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    by_type = defaultdict(list)
    for r in passed:
        by_type[r["task_type"]].append(r)
    for t in by_type:
        by_type[t].sort(key=lambda r: r["quality_score"], reverse=True)

    final = []
    deficits: Dict[str, int] = {}
    hot_displaced: List[Dict[str, Any]] = []

    for ttype, target in TARGET_RATIOS.items():
        bucket = by_type.get(ttype, [])
        if len(bucket) >= target:
            final.extend(bucket[:target])
            hot_displaced.extend(bucket[target:])
            deficits[ttype] = 0
        else:
            warm_for_type = sorted(
                [r for r in warm_reserve if r["task_type"] == ttype],
                key=lambda r: r["quality_score"], reverse=True,
            )
            need = target - len(bucket)
            accepted_hashes = {_row_hash(r) for r in bucket}
            fill = []
            for r in warm_for_type:
                if len(fill) >= need:
                    break
                h = _row_hash(r)
                if h in accepted_hashes:
                    continue
                accepted_hashes.add(h)
                fill.append(r)
            final.extend(bucket)
            final.extend(fill)
            deficits[ttype] = max(0, need - len(fill))

    for ttype, d in deficits.items():
        if d > 0:
            log(f"[GATE G4] WARNING: {ttype} short by {d} pairs. NOT padding (rule #9).")

    gate_log("G4", "REBALANCE", len(final), 0,
             f"targets={TARGET_RATIOS} | deficits={deficits} | hot_displaced={len(hot_displaced)}")
    return final, deficits


# ---------------------------------------------------------------------------
# CELL 11 - Gate G5: 5-layer leakage audit (train vs eval)
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int) -> set:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n:
        return set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def gate_g5_leakage(train: List[Dict[str, Any]],
                    evalset: List[Dict[str, Any]]
                    ) -> Tuple[List[Dict[str, Any]], int]:
    log("[GATE G5] Loading MiniLM for leakage audit...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # L1 parent-doc lock
    eval_parent_keys = {(r["source"], r["input"][:300]) for r in evalset if r["input"]}

    # L2 entity bleed: shared capitalized 3-grams
    eval_entity_set: set = set()
    for r in evalset:
        text = r["instruction"] + " " + r["input"] + " " + r["response"]
        caps3 = re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
        eval_entity_set.update(caps3)

    # L3 13-gram containment
    eval_ngrams: set = set()
    for r in evalset:
        eval_ngrams |= _ngrams(r["instruction"] + " " + r["response"], NGRAM_LEAK_SIZE)

    # L4 + L5 embedding cosine
    eval_texts  = [(r["instruction"] + " " + r["response"])[:1500] for r in evalset]
    train_texts = [(r["instruction"] + " " + r["response"])[:1500] for r in train]
    eval_emb  = _embed_texts(model, eval_texts)  if eval_texts  else np.zeros((0, 384))
    train_emb = _embed_texts(model, train_texts) if train_texts else np.zeros((0, 384))

    sim_max = np.zeros(len(train), dtype=np.float32)
    if eval_emb.shape[0] > 0 and train_emb.shape[0] > 0:
        CHUNK = 256
        for i in range(0, len(train), CHUNK):
            block = train_emb[i:i + CHUNK]
            sims = block @ eval_emb.T
            if sims.size:
                sim_max[i:i + CHUNK] = sims.max(axis=1)

    kept = []
    flagged = 0
    for idx, r in enumerate(train):
        reason = None
        if r["input"] and (r["source"], r["input"][:300]) in eval_parent_keys:
            reason = "parent_doc"
        else:
            text = r["instruction"] + " " + r["input"] + " " + r["response"]
            caps3 = set(re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b", text))
            if caps3 and len(caps3 & eval_entity_set) >= 10:
                reason = "entity_bleed"
            elif eval_ngrams and (_ngrams(r["instruction"] + " " + r["response"], NGRAM_LEAK_SIZE) & eval_ngrams):
                reason = "ngram13"
            elif sim_max[idx] >= PARAPHRASE_LEAK_THRESHOLD:
                reason = "paraphrase"
            elif sim_max[idx] >= LEAKAGE_COSINE_THRESHOLD:
                reason = "response_template"
        if reason is None:
            kept.append(r)
        else:
            flagged += 1
    gate_log("G5", "LEAKAGE", len(kept), flagged,
             f"5-layer (parent/entity/13gram/{PARAPHRASE_LEAK_THRESHOLD}/{LEAKAGE_COSINE_THRESHOLD})")
    return kept, flagged


# ---------------------------------------------------------------------------
# CELL 12 - Gate G6: final validation + GATES_PASSED.v4 sentinel
# ---------------------------------------------------------------------------

def gate_g6_final(train: List[Dict[str, Any]],
                  evalset: List[Dict[str, Any]],
                  deficits: Dict[str, int]) -> bool:
    n = len(train)
    by_type = Counter(r["task_type"] for r in train)
    for ttype, target in TARGET_RATIOS.items():
        got = by_type.get(ttype, 0)
        if got < target:
            log(f"[GATE G6] {ttype} short: {got}/{target} (deficit reported in G4; no padding)")

    ok = True
    for r in train + evalset:
        for f in REQUIRED_FIELDS:
            if not r.get(f):
                log(f"[GATE G6] FAIL: row missing field {f}.")
                ok = False
                break
        if not ok:
            break

    if ok:
        with open(SENTINEL_PATH, "w") as fh:
            fh.write(f"GATES_PASSED v4 - train={n} eval={len(evalset)} deficits={deficits}\n")
        gate_log("G6", "FINAL", n, 0, f"SENTINEL written: {SENTINEL_PATH}")
    else:
        gate_log("G6", "FINAL", 0, n, "FAILED - sentinel NOT written")
    return ok


# ---------------------------------------------------------------------------
# CELL 13 - Save + upload
# ---------------------------------------------------------------------------

def save_jsonl(rows: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"[SAVE] {len(rows)} rows -> {path}")


def upload_to_hf(train_path: str, eval_path: str):
    if not os.path.exists(SENTINEL_PATH):
        log(f"[UPLOAD] ABORT: sentinel {SENTINEL_PATH} missing. Gates did not pass.")
        return
    if not HF_TOKEN:
        log("[UPLOAD] ABORT: no HF_TOKEN.")
        return
    api = HfApi()
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo=HF_TARGET_FILE,
        repo_id=HF_TARGET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj=eval_path,
        path_in_repo=HF_EVAL_FILE,
        repo_id=HF_TARGET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    log(f"[UPLOAD] {HF_TARGET_REPO}/{HF_TARGET_FILE}")
    log(f"[UPLOAD] {HF_TARGET_REPO}/{HF_EVAL_FILE}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Cells 3+4
    all_rows = normalize_and_cap()

    # Cell 5 - eval split FIRST (before any dedup/filtering)
    train_pool, eval_pool = stratified_eval_split(all_rows)

    # Cell 6 - G0 schema (both halves)
    train_pool = gate_g0_schema(train_pool)
    eval_pool  = gate_g0_schema(eval_pool)

    # Cell 7 - G1 exact hash (both halves)
    train_pool = gate_g1_exact(train_pool)
    eval_pool  = gate_g1_exact(eval_pool)

    # Cell 8 - G2 quality scoring (train); eval scored for visibility only
    train_passed, train_warm, _train_cold = gate_g2_quality(train_pool)
    for r in eval_pool:
        scorer = _SCORERS.get(r["task_type"])
        r["quality_score"] = scorer(r) if scorer else 0

    # Cell 9 - G3 semantic dedup (sort by quality DESC before dedup)
    train_dedup = gate_g3_semantic(train_passed)

    # Cell 10 - G4 rebalance to targets
    train_final, deficits = gate_g4_rebalance(train_dedup, train_warm)

    # Cell 11 - G5 leakage audit
    train_final, n_leak = gate_g5_leakage(train_final, eval_pool)

    # Cell 12 - G6 final validation + sentinel
    ok = gate_g6_final(train_final, eval_pool, deficits)

    # Save
    save_jsonl(train_final, OUTPUT_PATH)
    save_jsonl(eval_pool, EVAL_OUTPUT_PATH)

    # Final dashboard
    by_type = Counter(r["task_type"] for r in train_final)
    total = len(train_final)
    log("")
    log(f"[COMPLETE] Total: {total:,} pairs")
    for ttype in ["structured_to_guidance", "narrative_to_facts",
                  "contract_to_clauses", "draft_document_section"]:
        n = by_type.get(ttype, 0)
        pct = (n / total * 100) if total else 0.0
        log(f"  {ttype}: {n:,} ({pct:.1f}%)")
    log(f"  eval_pool: {len(eval_pool):,}")
    log(f"  leakage_flagged: {n_leak}")
    log(f"  elapsed: {time.time() - t_start:.1f}s")

    # Cell 13 - upload (only if sentinel present)
    if ok:
        upload_to_hf(OUTPUT_PATH, EVAL_OUTPUT_PATH)
    else:
        log("[UPLOAD] skipped - gates failed")


if __name__ == "__main__":
    main()
