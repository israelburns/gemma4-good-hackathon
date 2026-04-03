# Jeremy AI x Gemma 4 — Live Demo (Kaggle Notebook)
# Gemma 4 Good Hackathon | Public inference demo
# Loads fine-tuned Gemma 4 E4B adapter and runs interactive legal guidance

import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.15.0", "accelerate>=0.35.0", "bitsandbytes>=0.45.0",
    "huggingface_hub", "gradio"])

import torch, os, time

print("=" * 60)
print("JEREMY AI x GEMMA 4 — Live Demo")
print("Access to Justice for All")
print("=" * 60)

# ── Load model with adapter ─────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_MODEL = "google/gemma-4-E4B-it"
ADAPTER = "israelburns/jeremy-gemma4"

print(f"Loading {BASE_MODEL} + adapter...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER, token=HF_TOKEN)
model.eval()
print("Model loaded.")

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are Jeremy, a legal procedural guidance AI for the Pro Se Network. "
    "You help self-represented litigants understand legal procedures, review contracts, "
    "and extract structured facts from their situations. You are NOT a lawyer. You provide "
    "procedural guidance only, never legal advice. Always include a disclaimer. "
    "When extracting facts, return structured JSON. When giving guidance, use plain English."
)

# ── Inference function ───────────────────────────────────────────────
def ask_jeremy(user_message, history=None):
    """Send a message to Jeremy and get guidance back."""
    messages = [{"role": "system", "content": SYSTEM_MSG}]

    # Add conversation history if provided
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": user_message})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response

# ── Gradio UI ────────────────────────────────────────────────────────
import gradio as gr

EXAMPLES = [
    "I was served with a lawsuit in New York for breach of contract. I have 20 days to respond. What do I do?",
    "My landlord is trying to evict me in California but didn't give proper notice. What are my rights?",
    "I want to file an EEOC complaint for workplace discrimination. Walk me through the process.",
    "I'm being sued for $8,000 in small claims court in Texas. How do I prepare?",
    "My employer fired me after I reported safety violations. Is this retaliation?",
]

demo = gr.ChatInterface(
    fn=ask_jeremy,
    title="Jeremy AI — Pro Se Legal Guidance",
    description=(
        "**Access to Justice for All** | Powered by Gemma 4\n\n"
        "Jeremy helps self-represented litigants navigate the legal system. "
        "He provides procedural guidance — not legal advice. "
        "80-phase deterministic engine covering 12 areas of law across 13 U.S. jurisdictions.\n\n"
        "*Jeremy is NOT a lawyer. Always consult an attorney for legal advice.*"
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
)

demo.launch(share=True)
