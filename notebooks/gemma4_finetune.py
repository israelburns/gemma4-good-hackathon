# Jeremy AI x Gemma 4 — LoRA Fine-Tune (Kaggle GPU)
# Gemma 4 Good Hackathon | Deadline: May 18, 2026
# Training Data: 5,196 legal Q&A pairs from Pro Se Network
# Model: google/gemma-4-E4B-it (4.5B effective, 8B total)
# Approach: QLoRA (4-bit) + LoRA rank 16
# Cost: $0.00 (Kaggle free GPU)

import subprocess, sys

# Install PyTorch with sm_60 (P100) support first — current Kaggle PyTorch drops sm_60
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
    "--index-url", "https://download.pytorch.org/whl/cu118"])

# Install deps — transformers from source required for gemma4 model type
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/huggingface/transformers.git",
    "peft>=0.15.0", "accelerate>=0.35.0", "bitsandbytes>=0.41.0",
    "trl>=0.16.0", "huggingface_hub"])

import torch, json, time, os

print("=" * 60)
print("JEREMY AI x GEMMA 4 — LoRA Fine-Tune")
print("Gemma 4 Good Hackathon")
print("=" * 60)
start = time.time()

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f} GB)")

# ── 1. Download training data from HuggingFace ──────────────────────
from huggingface_hub import hf_hub_download

print(f"\n[{time.strftime('%H:%M:%S')}] Downloading training data...")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

data_path = hf_hub_download(
    repo_id="israelburns/jeremy-training-data",
    filename="jeremy_training_v2.jsonl",
    repo_type="dataset",
    token=HF_TOKEN,
)

pairs = []
with open(data_path) as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))
print(f"  Loaded: {len(pairs)} pairs")

# ── 2. Format for Gemma 4 chat template ─────────────────────────────
SYSTEM_MSG = (
    "You are Jeremy, a legal procedural guidance AI for the Pro Se Network. "
    "You help self-represented litigants understand legal procedures, review contracts, "
    "and extract structured facts from their situations. You are NOT a lawyer. You provide "
    "procedural guidance only, never legal advice. Always include a disclaimer. "
    "When extracting facts, return structured JSON. When giving guidance, use plain English."
)

def format_pair(pair):
    """Format training pair into Gemma 4 chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
    }

formatted = [format_pair(p) for p in pairs]

# Split 95/5
import random
random.seed(42)
random.shuffle(formatted)
split_idx = int(len(formatted) * 0.95)
train_data = formatted[:split_idx]
eval_data = formatted[split_idx:]
print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

# ── 3. Load Gemma 4 E4B-it with 4-bit quantization ──────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "google/gemma-4-E4B-it"

# Detect GPU capability — P100 (sm_60) can't run bitsandbytes 4-bit
cuda_cap = torch.cuda.get_device_capability(0)
use_4bit = cuda_cap[0] >= 7  # sm_70+ required for bitsandbytes NF4
print(f"\n[{time.strftime('%H:%M:%S')}] Loading {MODEL_ID} ({'4-bit NF4' if use_4bit else 'fp16 (old GPU)'})...")
print(f"  CUDA capability: sm_{cuda_cap[0]}{cuda_cap[1]} — 4-bit: {use_4bit}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

if use_4bit:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        dtype=torch.bfloat16,
    )
else:
    # fp16 fallback for P100/older GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        token=HF_TOKEN,
        dtype=torch.float16,
    )

model = prepare_model_for_kbit_training(model)
print(f"  Model loaded. Trainable params before LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── 4. Apply LoRA ───────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  LoRA applied. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── 5. Tokenize dataset ─────────────────────────────────────────────
from datasets import Dataset

def tokenize_messages(example):
    """Apply Gemma 4 chat template and tokenize."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(text, truncation=True, max_length=2048, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print(f"\n[{time.strftime('%H:%M:%S')}] Tokenizing...")
train_dataset = Dataset.from_list(train_data).map(tokenize_messages, remove_columns=["messages"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_messages, remove_columns=["messages"])
print(f"  Train tokens: {sum(len(x) for x in train_dataset['input_ids']):,}")
print(f"  Eval tokens: {sum(len(x) for x in eval_dataset['input_ids']):,}")

# ── 6. Train ────────────────────────────────────────────────────────
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

OUTPUT_DIR = "/kaggle/working/jeremy-gemma4"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=use_4bit,       # bf16 only on sm_70+ (T4/A100); fp16 on P100
    fp16=not use_4bit,
    optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"\n[{time.strftime('%H:%M:%S')}] Training...")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch: {training_args.per_device_train_batch_size} x {training_args.gradient_accumulation_steps} accum = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps} effective")
print(f"  LR: {training_args.learning_rate}")

trainer.train()

elapsed = time.time() - start
print(f"\n[{time.strftime('%H:%M:%S')}] Training complete in {elapsed/60:.1f} minutes")

# ── 7. Eval loss ────────────────────────────────────────────────────
eval_results = trainer.evaluate()
print(f"  Eval loss: {eval_results['eval_loss']:.4f}")

# ── 8. Save adapter + push to HuggingFace ───────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Saving adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")

if HF_TOKEN:
    print("  Pushing to HuggingFace...")
    model.push_to_hub("israelburns/jeremy-gemma4", token=HF_TOKEN)
    tokenizer.push_to_hub("israelburns/jeremy-gemma4", token=HF_TOKEN)
    print("  Pushed: huggingface.co/israelburns/jeremy-gemma4")

# ── 9. Quick inference test ──────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Testing inference...")
test_prompt = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": "I was served with a lawsuit in New York for breach of contract. I have 20 days to respond. What do I do?"},
]

input_text = tokenizer.apply_chat_template(test_prompt, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n--- TEST RESPONSE ---\n{response}\n--- END ---")

total_time = time.time() - start
print(f"\n{'='*60}")
print(f"COMPLETE | Total time: {total_time/60:.1f} min | Eval loss: {eval_results['eval_loss']:.4f}")
print(f"Adapter: {OUTPUT_DIR}/adapter")
print(f"{'='*60}")
