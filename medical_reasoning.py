# medical_reasoning.py
# ─────────────────────────────────────────────
# Medical Q&A using HuggingFace model trained on
# FreedomIntelligence/medical-o1-reasoning-SFT data
# ─────────────────────────────────────────────
# SETUP:
# 1. Create a .env file in this folder with: HF_TOKEN=hf_your_token_here
# 2. pip install transformers torch python-dotenv accelerate
# 3. python medical_reasoning.py

import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ─────────────────────────────────────────────
# Load token securely from .env file
# NEVER hardcode your token in code
# ─────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found! "
        "Create a .env file with: HF_TOKEN=hf_your_token_here"
    )

# ─────────────────────────────────────────────
# Model Selection
# We use FreedomIntelligence/Apollo-7B which is
# trained on the medical-o1-reasoning-SFT dataset
# you linked — so it directly uses that data!
#
# Why Apollo-7B?
# - Trained on medical reasoning data
# - Open access on Hugging Face
# - Good balance of quality vs size
# - 7B parameters — runs on free Colab T4 GPU
# ─────────────────────────────────────────────
MODEL_NAME = "FreedomIntelligence/Apollo-7B"

print(f"Loading model: {MODEL_NAME}")
print("This may take a few minutes on first run...")

# ─────────────────────────────────────────────
# Load tokenizer and model
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",          # automatically uses GPU if available
    trust_remote_code=True
)

print(f"Model loaded successfully!")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")


# ─────────────────────────────────────────────
# Create inference pipeline
# ─────────────────────────────────────────────
medical_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


# ─────────────────────────────────────────────
# Query function
# ─────────────────────────────────────────────
def ask_medical_question(question: str, max_new_tokens: int = 300) -> str:
    """
    Ask a medical question and get a reasoned response.
    
    Args:
        question: Medical question to ask
        max_new_tokens: Maximum length of response
    
    Returns:
        Model's response as a string
    """
    # Format prompt for medical reasoning
    prompt = f"""You are a medical expert. Answer the following medical question 
with clear reasoning and evidence-based information.

Question: {question}

Answer:"""

    response = medical_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,        # lower = more focused/accurate
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract just the answer part (remove the prompt)
    full_output = response[0]["generated_text"]
    answer = full_output.split("Answer:")[-1].strip()
    return answer


# ─────────────────────────────────────────────
# Run example queries
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What are the symptoms of Type 2 diabetes?",
        "What is the first-line treatment for hypertension?",
        "How does HIPAA protect patient information?",
        "What are the social determinants of health?"
    ]

    print("\n" + "="*60)
    print("MEDICAL REASONING MODEL — QUERY RESULTS")
    print("="*60)

    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 40)
        answer = ask_medical_question(question)
        print(f"A: {answer}")
        print("="*60)