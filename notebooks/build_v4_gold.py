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

# v3_gold 'area' values that are genuinely on-domain for a divorce assistant.
_V3_FAMILY_AREAS = {"custody", "family_law", "child_support", "domestic_violence",
                    "matrimonial", "divorce"}

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
        # Tag family-law-area rows as a distinct, higher-priority source so they
        # win narrative/structured slots over employment / police-misconduct v3.
        area = (r.get("area") or "").lower()
        src = "v3_gold_family" if area in _V3_FAMILY_AREAS else "v3_gold"
        rows.append({
            "task_type":    bucket,
            "instruction":  (r.get("instruction") or "").strip(),
            "input":        "",
            "response":     (r.get("response") or "").strip(),
            "source":       src,
            "jurisdiction": r.get("jurisdiction", ""),
        })
    fam = sum(1 for r in rows if r["source"] == "v3_gold_family")
    log(f"v3_gold: {len(rows)} rows ({fam} family-area, prioritized)")
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

# ─────────────────────────────────────────────────────────────────────────────
# DIVORCE-NATIVE SOURCE — the actual NY uncontested-divorce UD workflow that
# ships inside the Jeremy project (pro-se-network/src). Authoritative,
# hand-authored, NY-specific. This is the on-domain signal the HF datasets
# could not provide (Kimi Stage 5 BLOCKER 3 fix).
# ─────────────────────────────────────────────────────────────────────────────

DIVORCE_SRC = "/Users/adam15obong/Documents/Money_Codes/pro-se-network/src"

def _flatten(v, depth=0):
    """Render a nested str/list/dict value as readable prose."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, list):
        parts = [_flatten(x, depth + 1) for x in v]
        return "\n".join(f"- {p}" for p in parts if p)
    if isinstance(v, dict):
        out = []
        for k, val in v.items():
            t = _flatten(val, depth + 1)
            if t:
                label = str(k).replace("_", " ")
                out.append(f"{label}: {t}" if "\n" not in t else f"{label}:\n{t}")
        return "\n".join(out)
    return ""

# Per-structure instruction framing — keeps the generated question on-domain
# and specific instead of a generic "explain this".
_DIVORCE_STRUCTS = {
    # module, attr, title_keys, instruction_template
    ("divorce_workflow", "UD_FORMS"):          ("In a New York uncontested divorce, what is form {title} and when is it filed?"),
    ("divorce_workflow", "ORDER_OF_OPERATIONS"):("In a New York uncontested divorce, what does this step involve: {title}?"),
    ("divorce_workflow", "EDGE_CASES"):        ("While filing a New York uncontested divorce pro se: {title}. What should I do?"),
    ("divorce_workflow", "STOP_TRIGGERS"):     ("When filing for divorce pro se in New York, should I stop and consult an attorney if: {title}?"),
    ("divorce_workflow", "CHECKLISTS"):        ("What belongs on the {title} checklist for a New York uncontested divorce?"),
    ("divorce_workflow", "CRITICAL_WARNINGS"): ("What is a critical warning I must know when filing a New York uncontested divorce pro se?"),
    ("divorce_workflow", "INTAKE_FIELDS"):     ("What information do I need to provide for the '{title}' field when preparing New York divorce paperwork, and why?"),
    ("divorce_basics",   "DIVORCE_TYPES"):     ("Explain the '{title}' option for someone deciding how to approach their divorce."),
    ("divorce_basics",   "PROPERTY_DIVISION"): ("How does '{title}' work when dividing property in a divorce?"),
    ("divorce_basics",   "ALIMONY_TYPES"):     ("Explain '{title}' alimony — what it is and when a court awards it."),
    ("divorce_basics",   "STATE_PROCEDURES"):  ("What is the divorce procedure in {title}?"),
    ("default_judgment_guide", "DEFAULT_TYPES"):     ("In civil litigation, explain '{title}' — what it is and how it works."),
    ("default_judgment_guide", "DEFAULT_PROCEDURES"):("What is the step-by-step default-judgment procedure for the {title}?"),
    ("default_judgment_guide", "VACATING_GROUNDS"):  ("Can a default judgment be vacated on the ground of '{title}'? Explain."),
    ("default_judgment_guide", "DEFAULT_PREVENTION_TIPS"):("How can a defendant avoid having a default judgment entered against them?"),
    ("judgment_collection", "COLLECTION_METHODS"):   ("How does the '{title}' method of collecting on a judgment work?"),
}

def pull_divorce_native():
    import importlib
    if DIVORCE_SRC not in os.sys.path:
        os.sys.path.insert(0, DIVORCE_SRC)
    rows = []
    for (modname, attr), instr_tmpl in _DIVORCE_STRUCTS.items():
        try:
            mod = importlib.import_module(modname)
            struct = getattr(mod, attr)
        except Exception as e:
            log(f"  divorce_native skip {modname}.{attr}: {str(e)[:60]}")
            continue
        # normalize to a list of (title, value) entries
        entries = []
        if isinstance(struct, dict):
            for k, v in struct.items():
                title = (v.get("name") or v.get("label") or v.get("title")
                         or v.get("category") or str(k).replace("_", " ")) if isinstance(v, dict) else str(k)
                entries.append((title, v))
        elif isinstance(struct, list):
            for item in struct:
                if isinstance(item, dict):
                    title = (item.get("name") or item.get("label") or item.get("title")
                             or item.get("form_id") or item.get("scenario")
                             or (f"step {item.get('step')} — {item.get('summary','')}"
                                 if item.get("step") else "") or "")
                else:
                    title = ""
                entries.append((title, item))
        for title, value in entries:
            body = _flatten(value)
            if len(body) < 150:
                continue
            instr = instr_tmpl.format(title=str(title).strip()[:120]) if "{title}" in instr_tmpl else instr_tmpl
            rows.append({
                "task_type":    "structured_to_guidance",
                "instruction":  instr,
                "input":        "",
                "response":     ("This is procedural legal information for New York, "
                                 "not legal advice.\n\n" + body),
                "source":       f"divorce_native:{attr}",
                "jurisdiction": "NY",
            })
    log(f"divorce_native: {len(rows)} rows (NY uncontested-divorce UD workflow)")
    return rows

def pull_ud_documents():
    """Run the project's own NY UD-1..UD-12 document generators over a spread of
    synthetic uncontested-divorce cases. Each (case x form) is a real, correctly
    structured NY divorce document — authoritative draft_document_section signal.
    Legal forms ARE templated; these are intentionally exempt from near-dedup."""
    import importlib
    if DIVORCE_SRC not in os.sys.path:
        os.sys.path.insert(0, DIVORCE_SRC)
    try:
        udt = importlib.import_module("ud_document_templates")
    except Exception as e:
        log(f"  ud_documents skip: {str(e)[:80]}")
        return []

    first = ["Maria","James","Aisha","Robert","Linda","Carlos","Wei","Sarah","David","Nadia",
             "Michael","Elena","Thomas","Grace","Omar","Patricia","Kevin","Rosa","Daniel","Joyce",
             "Andre","Fatima","Steven","Camille","Jonah"]
    last  = ["Rivera","Thompson","Okafor","Chen","Walsh","Delgado","Park","Hughes","Bennett","Haddad",
             "Russo","Petrov","Coleman","Nguyen","Bauer","Flores","Sullivan","Marino","Brooks","Kim",
             "Foster","Abboud","Greene","Laurent","Stein"]
    streets = ["12 Bay Street","88 Victory Boulevard","240 Forest Avenue","57 Richmond Terrace",
               "910 Castleton Avenue","33 Hyatt Street","145 Targee Street","76 St Marks Place",
               "501 Port Richmond Avenue","219 Bard Avenue"]
    mplaces = ["Staten Island, NY","Manhattan, NY","Brooklyn, NY","Newark, NJ","Philadelphia, PA",
               "Jersey City, NJ","Queens, NY","Yonkers, NY"]
    res_bases = ["plaintiff_richmond","defendant_richmond","married_in_ny",
                 "both_ny_resident","grounds_in_ny","two_year"]
    zips = ["10301","10302","10303","10304","10305","10306","10310","10314"]

    rows = []
    for i in range(25):
        pf, pl = first[i], last[i]
        df, dl = first[(i + 7) % 25], last[(i + 13) % 25]
        data = {
            "plaintiff_name":    f"{pf} {pl}",
            "plaintiff_address": f"{streets[i % len(streets)]}, Staten Island, NY {zips[i % len(zips)]}",
            "defendant_name":    f"{df} {dl}",
            "defendant_address": f"{streets[(i + 5) % len(streets)]}, Staten Island, NY {zips[(i + 3) % len(zips)]}",
            "marriage_date":     f"{2005 + (i % 15)}-{1 + (i % 9):02d}-{1 + (i % 27):02d}",
            "marriage_place":    mplaces[i % len(mplaces)],
            "irretrievable_breakdown": True, "no_children": True,
            "no_property": True, "both_cooperate": True,
            "residency_basis":   res_bases[i % 6],
            "service_method":    "process_server" if i % 2 else "personal_nonparty",
            "notary_plan":       "separate_notary",
            "index_number":      f"{150000 + i * 37}/2026",
            "plaintiff_county":  "Richmond", "defendant_county": "Richmond",
            "server_name":       f"{first[(i + 3) % 25]} {last[(i + 9) % 25]}",
            "server_address":    f"{streets[(i + 2) % len(streets)]}, Staten Island, NY {zips[(i + 1) % len(zips)]}",
            "service_date":      f"2026-{1 + (i % 9):02d}-{2 + (i % 26):02d}",
            "service_time":      f"{9 + (i % 8)}:00 AM",
            "service_location":  f"{streets[(i + 5) % len(streets)]}, Staten Island, NY {zips[(i + 3) % len(zips)]}",
            "defendant_description": "as identified at the address of service",
        }
        try:
            result = udt.generate_ud_documents(data)
        except Exception as e:
            log(f"  ud_documents case {i} failed: {str(e)[:60]}")
            continue
        facts = (f"Plaintiff {data['plaintiff_name']} v. Defendant {data['defendant_name']}; "
                 f"Richmond County, New York; married {data['marriage_date']} in "
                 f"{data['marriage_place']}; no children under 21; no marital property or "
                 f"maintenance; no-fault grounds (irretrievable breakdown, DRL §170(7)).")
        for doc in result.get("documents", []):
            if not doc.get("success") or len(doc.get("full_text", "")) < 150:
                continue
            rows.append({
                "task_type":    "draft_document_section",
                "instruction":  (f"Draft a {doc['form_id']} ({doc['title']}) for a New York "
                                 f"uncontested divorce. Case facts: {facts}"),
                "input":        "",
                "response":     doc["full_text"],
                "source":       f"ud_template:{doc['form_id']}",
                "jurisdiction": "NY",
            })
    log(f"ud_documents: {len(rows)} rows (synthetic NY divorce UD drafts, dedup-exempt)")
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

# Source priority — lower = kept first. On-domain NY divorce content (the UD
# workflow + generated UD documents) wins every slot it can; v3_gold (curated
# Jeremy data) next; generic legal last.
_SRC_PRIORITY = {"divorce_native": 0, "ud_template": 0, "v3_gold_family": 0, "v3_gold": 1}
def _priority(r):
    return _SRC_PRIORITY.get(r["source"].split(":")[0], 2)

# Legal document templates are SUPPOSED to be near-identical across cases —
# templated repetition with varied field-fills IS the drafting signal. Exempt
# them from near-dedup (they still pass MD5 exact dedup).
_DEDUP_EXEMPT = {"ud_template"}

def semantic_dedup_and_slice(rows):
    final = []
    for bucket in BUCKETS:
        pool = [r for r in rows if r["task_type"] == bucket]
        # priority first, then richest response — divorce content is never
        # displaced by longer-but-generic legal text
        pool.sort(key=lambda r: (_priority(r), -len(r["response"])))
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
            src = r["source"].split(":")[0]
            if src in _DEDUP_EXEMPT:
                # legal templates: templated repetition is the signal — keep
                kept.append(r); kept_grams.append(_ngrams(r["response"]))
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
    raw += pull_divorce_native()   # on-domain NY divorce UD workflow guidance
    raw += pull_ud_documents()     # on-domain NY divorce UD-1..UD-12 drafts
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
