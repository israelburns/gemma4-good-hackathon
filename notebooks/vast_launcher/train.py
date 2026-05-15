#!/usr/bin/env python3
# Jeremy AI x Gemma 4 — Fine-Tune v4 (GOLD / Ace Burns Protocol build)
# VM-side training script. HF_TOKEN comes from env (SSH-injected, not on disk).
#
# FIXES (Ace Burns Protocol — Stage 2 Kimi + Stage 3 Gemini):
# - UNSLOTH_USE_MODELSCOPE=1 — bypasses Unsloth _get_statistics HF-timeout
# - import torch._inductor.config explicitly — unsloth_zoo's
#   inspect.getsource(torch._inductor.config) needs the submodule bound
# - Triton >=3.1 — 3.0 (PyTorch 2.4.1 default) does NOT support sm_100 Blackwell;
#   without this, B200/RTX 5090 hard-crash at training step 1
# - bitsandbytes >=0.45 + accelerate >=0.34 — Blackwell sm_100 kernels live there
# - Install order: dep bumps FIRST, then unsloth (so it sees the new versions)
# - signal.SIGALRM(180) bound around HF push only (not the whole script)

import os, sys, json, time, random, signal, subprocess

HF_TOKEN = os.environ["HF_TOKEN"]   # required, fail loud if missing
# Run #6 found: ModelScope download from Vast hosts is glacial (141 KB/s).
# HF Hub is much faster on global routes. Drop the ModelScope env var; we
# pre-download the model below via huggingface_hub.snapshot_download() so
# unsloth's own telemetry path (the original reason we used MODELSCOPE)
# never fires — model is cached on disk before unsloth.from_pretrained().
os.environ["UNSLOTH_OFFLINE"]           = "1"   # Gemini S3b: block runtime .so downloads
os.environ["PYTORCH_CUDA_ALLOC_CONF"]   = "expandable_segments:True"
os.environ["BITSANDBYTES_NOWELCOME"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"]    = "false"
os.environ["CUDA_VISIBLE_DEVICES"]      = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["WANDB_DISABLED"]            = "true"

print("=== INSTALL DEPS (Ada sm_89 — let unsloth pick its tested deps) ===", flush=True)
# Run #9 finding: bnb>=0.45 has a C-symbol API mismatch with unsloth 2026.5.2
# (`lib.cdequantize_blockwise_fp32` was renamed/removed). All the Blackwell
# pins (torch 2.11 trident, triton 3.1, bnb 0.45 floor, torchao 0.7) are
# IRRELEVANT on Ada sm_89. Strip them all — install unsloth WITH its declared
# deps (no --no-deps) so pip resolves the bnb/triton/accelerate trio that
# unsloth was actually tested against.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "--upgrade", "--no-cache-dir",
    # Kimi R2: pin unsloth + unsloth_zoo to stop them force-upgrading the image's
    # torch 2.5.1 / triton stack mid-resolve. These versions were verified to
    # import cleanly in Run #8 + Run #9.
    "unsloth==2026.5.2", "unsloth_zoo==2026.5.1",
    # Kimi S5: bnb 0.43.3 is the known-good. 0.45+ removed the
    # cdequantize_blockwise_fp32 C symbol unsloth's internal code still calls.
    "bitsandbytes==0.43.3",
    # Downstream stack unsloth doesn't auto-pull
    "transformers>=4.51.0,!=5.0.0,!=5.1.0",
    "trl>=0.23.0,<=0.24.0",
    "datasets>=2.18.0,<4.4.0",
    "peft>=0.15.0",
    "huggingface_hub>=0.30.0",
    "hf-transfer",
    # xformers DROPPED — it's optional for unsloth (SDPA fallback works on Ada)
    # and unpinned xformers wheel could fight the image's torch 2.5.1+cu124
    "tyro", "ninja",
])

# Forensic: dump installed versions to log so a silent downgrade is at least visible
print("=== PIP FREEZE (forensics) ===", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "freeze"])

# Gemini S3b BULLETPROOF MONKEY-PATCH — neutralize the inspect.getsource crash
# at the source. `import torch._inductor.config` alone may fail when config is
# loaded as a compiled C extension (no .py source), causing inspect.getsource()
# to throw TypeError. Override inspect.getsource for that one path.
import inspect as _inspect
_orig_getsource = _inspect.getsource
def _safe_getsource(obj):
    try:
        if "torch._inductor.config" in str(obj):
            return ""
    except Exception:
        pass
    return _orig_getsource(obj)
_inspect.getsource = _safe_getsource

# Belt-and-braces: also bind the submodule so any code path that needs the
# attribute (not the source) still works.
import torch
import torch._inductor.config   # noqa: F401

# NOW it's safe to import unsloth
from unsloth import FastLanguageModel
from datasets import Dataset
from huggingface_hub import hf_hub_download
from trl import SFTConfig, SFTTrainer

print("=" * 60); print("JEREMY AI x GEMMA 4 — Unsloth QLoRA (GOLD)"); print("=" * 60)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {torch.cuda.get_device_name(i)} ({p.total_memory/1e9:.1f} GB)", flush=True)
start = time.time()

print("=== LOAD DATA ===", flush=True)
data_path = hf_hub_download(repo_id="israelburns/jeremy-training-data",
                            filename="jeremy_training_v4_gold.jsonl",
                            repo_type="dataset", token=HF_TOKEN)
with open(data_path, "r", encoding="utf-8") as f:
    pairs = [json.loads(l) for l in f if l.strip()]
print(f"Loaded: {len(pairs)} pairs", flush=True)

SYSTEM_MSG = ("You are Jeremy, a legal procedural guidance AI for the Pro Se Network. "
              "You help self-represented litigants understand legal procedures, review contracts, "
              "and extract structured facts. You are NOT a lawyer. Procedural guidance only, never "
              "legal advice. Always include a disclaimer.")
formatted = [{"messages": [
    {"role": "system",    "content": SYSTEM_MSG},
    {"role": "user",      "content": p["instruction"]},
    {"role": "assistant", "content": p["response"]},
]} for p in pairs]
random.seed(42); random.shuffle(formatted)
split = max(1, int(len(formatted) * 0.95))
train_data, eval_data = formatted[:split], formatted[split:]
print(f"Train: {len(train_data)} | Eval: {len(eval_data)}", flush=True)

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
MAX_SEQ  = 2048
OUTPUT_DIR = "/root/jeremy-gemma4"

# N=12 parallel sweep — cell-specific hyperparams + distinct HF push target.
# Each cell pushes to: israelburns/jeremy-gemma4-{CELL_ID}
# Default values match the V2 baseline (cell B2).
CELL_ID  = os.environ.get("CELL_ID", "default")
LORA_R   = int(os.environ.get("LORA_R", "16"))
LORA_A   = int(os.environ.get("LORA_A", str(LORA_R * 2)))
LR       = float(os.environ.get("LR", "1e-4"))
HF_REPO  = f"israelburns/jeremy-gemma4-{CELL_ID}"
print(f"=== CELL CONFIG: id={CELL_ID} r={LORA_R} alpha={LORA_A} lr={LR} repo={HF_REPO}", flush=True)

# Pre-download from HF Hub directly. Bypasses unsloth's telemetry/source-routing
# logic entirely and avoids the slow ModelScope CDN. HF is fast on Vast hosts.
print("=== PRE-DOWNLOAD MODEL via HF Hub ===", flush=True)
from huggingface_hub import snapshot_download
LOCAL_MODEL = "/root/gemma4-model"
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_MODEL,
    token=HF_TOKEN,
    max_workers=8,                # parallelize the 3 shards
)
print(f"  model cached at {LOCAL_MODEL}", flush=True)

print("=== LOAD MODEL (from local cache) ===", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LOCAL_MODEL, max_seq_length=MAX_SEQ, dtype=torch.float16,
    load_in_4bit=True, token=HF_TOKEN)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_A, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)
model.print_trainable_parameters()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

def render_chat(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

print("=== FORMAT ===", flush=True)
train_ds = Dataset.from_list(train_data).map(render_chat, remove_columns=["messages"])
eval_ds  = Dataset.from_list(eval_data ).map(render_chat, remove_columns=["messages"])

print("=== TRAIN ===", flush=True)
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    learning_rate=LR, warmup_steps=30, weight_decay=0.05,
    logging_steps=25, eval_strategy="steps", eval_steps=100,
    save_strategy="steps", save_steps=200, save_total_limit=2,
    fp16=True, bf16=False, optim="adamw_8bit",
    lr_scheduler_type="cosine", max_grad_norm=0.3, report_to="none",
    dataloader_pin_memory=False, dataloader_num_workers=0,
    remove_unused_columns=False, seed=42,
    max_seq_length=MAX_SEQ, packing=False, dataset_text_field="text")
trainer = SFTTrainer(model=model, args=args,
                     train_dataset=train_ds, eval_dataset=eval_ds,
                     processing_class=tokenizer)
torch.cuda.empty_cache()
trainer.train()

print("=== EVAL ===", flush=True)
try:
    er = trainer.evaluate()
    print(f"Eval loss: {er['eval_loss']:.4f}", flush=True)
except Exception as e:
    print(f"Eval skipped: {e}", flush=True)

adapter = f"{OUTPUT_DIR}/adapter"
model.save_pretrained(adapter)
tokenizer.save_pretrained(adapter)

# Gemini Stage 3 fix — bound HF push with SIGALRM, NOT timeout on whole script
print("=== HF PUSH (180s SIGALRM bound) ===", flush=True)
def _alrm(signum, frame):
    raise TimeoutError("HF push timed out at SIGALRM(180)")
signal.signal(signal.SIGALRM, _alrm)
signal.alarm(180)
try:
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print("Push complete.", flush=True)
except TimeoutError as e:
    print(f"!! HF push timed out: {e} — adapter saved locally at {adapter}", flush=True)
except Exception as e:
    print(f"!! HF push failed: {e}", flush=True)
finally:
    signal.alarm(0)

elapsed = int(time.time() - start)
print(f"=== COMPLETE — {elapsed//60}m {elapsed%60}s ===", flush=True)
print(f"HF: https://huggingface.co/{HF_REPO}", flush=True)
