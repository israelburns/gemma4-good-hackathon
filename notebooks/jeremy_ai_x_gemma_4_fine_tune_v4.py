import os
import sys
import json
import time
import random
import subprocess

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--upgrade", "--no-cache-dir",
    "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git",
])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.51.0,!=5.0.0,!=5.1.0",
    "trl>=0.23.0,<=0.24.0",
    "datasets>=2.18.0,<4.4.0",
    "peft>=0.15.0",
    "accelerate>=1.0.0",
    "bitsandbytes>=0.43.3",
    "huggingface_hub>=0.30.0",
    "hf-transfer",
])

from unsloth import FastLanguageModel  # must import before trl/transformers/peft
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from trl import SFTConfig, SFTTrainer

print("=" * 60)
print("JEREMY AI x GEMMA 4 — Unsloth QLoRA (T4 sm_75, fp16)")
print("=" * 60)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)")
start = time.time()

data_path = hf_hub_download(
    repo_id="israelburns/jeremy-training-data",
    filename="jeremy_training_v4_gold.jsonl",
    repo_type="dataset",
    token=HF_TOKEN,
)

with open(data_path, "r", encoding="utf-8") as f:
    pairs = [json.loads(line) for line in f if line.strip()]

print(f"Loaded: {len(pairs)} pairs")

SYSTEM_MSG = (
    "You are Jeremy, a legal procedural guidance AI for the Pro Se Network. "
    "You help self-represented litigants understand legal procedures, review contracts, "
    "and extract structured facts. You are NOT a lawyer. Procedural guidance only, never "
    "legal advice. Always include a disclaimer."
)

formatted = [{
    "messages": [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": p["instruction"]},
        {"role": "assistant", "content": p["response"]},
    ]
} for p in pairs]

random.seed(42)
random.shuffle(formatted)

split = max(1, int(len(formatted) * 0.95))
if split >= len(formatted):
    split = len(formatted) - 1

train_data = formatted[:split]
eval_data = formatted[split:]
print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

# Codex-verified model ID — unsloth/gemma-4-e4b-it-bnb-4bit does NOT exist
MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
MAX_SEQ = 1024
OUTPUT_DIR = "/kaggle/working/jeremy-gemma4"

print(f"\nLoading {MODEL_ID} via Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
    token=HF_TOKEN,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
model.print_trainable_parameters()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

def render_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("\nFormatting dataset...")
train_ds = Dataset.from_list(train_data).map(render_chat, remove_columns=["messages"])
eval_ds = Dataset.from_list(eval_data).map(render_chat, remove_columns=["messages"])

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,
    warmup_steps=30,
    weight_decay=0.05,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    seed=42,
    max_seq_length=MAX_SEQ,
    packing=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

print("\nTraining...")
torch.cuda.empty_cache()
trainer.train()

try:
    eval_results = trainer.evaluate()
    print(f"Eval loss: {eval_results['eval_loss']:.4f}")
except Exception as e:
    print(f"Eval skipped: {e}")
    eval_results = {}

adapter_dir = f"{OUTPUT_DIR}/adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print("Pushing to HuggingFace...")
model.push_to_hub("israelburns/jeremy-gemma4", token=HF_TOKEN)
tokenizer.push_to_hub("israelburns/jeremy-gemma4", token=HF_TOKEN)
print("Push complete.")

elapsed = int(time.time() - start)
print(f"\nCOMPLETE — {elapsed // 60}m {elapsed % 60}s")
print(f"Eval loss: {eval_results['eval_loss']:.4f}")
print("HF: https://huggingface.co/israelburns/jeremy-gemma4")
