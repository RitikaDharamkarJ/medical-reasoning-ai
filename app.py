import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import torch
import time

# ─────────────────────────────────────────────
# Load token
# ─────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env", override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HealthMind AI",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }

    .stApp {
        background: #f8f6f1;
    }

    /* Header */
    .header {
        background: linear-gradient(135deg, #1a3a4a 0%, #0d2233 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 24px 24px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,180,150,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header p {
        color: rgba(255,255,255,0.65);
        font-size: 1rem;
        margin: 0.5rem 0 0;
        font-weight: 300;
    }
    .header-badge {
        display: inline-block;
        background: rgba(0,180,150,0.2);
        border: 1px solid rgba(0,180,150,0.4);
        color: #00b496;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1a3a4a !important;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }
    [data-testid="stSidebar"] label {
        color: rgba(255,255,255,0.55) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stSidebar"] .stSelectbox div {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider {
        color: white !important;
    }

    /* Question input */
    .question-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .question-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #1a3a4a;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    .stTextArea textarea {
        background: #f8f6f1 !important;
        border: 1.5px solid #e0dbd0 !important;
        border-radius: 10px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 0.95rem !important;
        color: #1a3a4a !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #00b496 !important;
        box-shadow: 0 0 0 3px rgba(0,180,150,0.1) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #1a3a4a, #0d6e5c) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 2rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* Response card */
    .response-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
        border-left: 4px solid #00b496;
        margin-top: 1.5rem;
    }
    .response-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .response-tag {
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        font-weight: 600;
        color: #1a3a4a;
    }
    .response-text {
        color: #2d4a5a;
        font-size: 0.97rem;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    /* Example questions */
    .example-label {
        font-size: 0.8rem;
        color: #8a8070;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .example-btn button {
        background: white !important;
        color: #1a3a4a !important;
        border: 1.5px solid #e0dbd0 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.75rem !important;
        font-weight: 400 !important;
    }
    .example-btn button:hover {
        border-color: #00b496 !important;
        color: #00b496 !important;
        background: rgba(0,180,150,0.05) !important;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff8e6;
        border: 1px solid #f0d080;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #7a6020;
        margin-top: 1rem;
    }

    /* Stats */
    .stat-card {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #00b496 !important;
    }
    .stat-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.5) !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 0 !important; max-width: 1100px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""


# ─────────────────────────────────────────────
# Model loading (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name, hf_token):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return tokenizer, model, pipe


# ─────────────────────────────────────────────
# Inference function
# ─────────────────────────────────────────────
def ask_medical_question(pipe, tokenizer, question, max_new_tokens, temperature):
    prompt = f"""You are a medical expert. Answer the following medical question with clear reasoning and evidence-based information.

Question: {question}

Answer:"""

    response = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = response[0]["generated_text"]
    answer = full_output.split("Answer:")[-1].strip()
    return answer


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # API token
    sidebar_token = st.text_input(
        "Hugging Face Token",
        value=HF_TOKEN or "",
        type="password",
        help="Your HF token from huggingface.co/settings/tokens"
    )

    # Model selection
    model_choice = st.selectbox(
        "Model",
        [
            "FreedomIntelligence/Apollo-7B",
            "microsoft/BioGPT-Large",
            "medalpaca/medalpaca-7b",
        ],
        index=0
    )

    st.markdown("### 🎛️ Parameters")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.3, 0.05,
                            help="Lower = more precise, Higher = more creative")
    max_tokens = st.slider("Max Response Tokens", 100, 500, 300, 50)

    st.markdown("---")

    # Load model button
    if st.button("Load Model", use_container_width=True):
        if not sidebar_token:
            st.error("Please enter your HF token first")
        else:
            with st.spinner("Loading model... this takes a few minutes"):
                try:
                    tokenizer, model, pipe = load_model(model_choice, sidebar_token)
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.session_state.pipeline = pipe
                    st.success("Model loaded!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{st.session_state.total_queries}</div>
            <div class="stat-label">Queries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{len(st.session_state.history)}</div>
            <div class="stat-label">History</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div style="font-size:0.75rem; color:rgba(255,255,255,0.4); line-height:1.6;">
        Model: {model_choice.split('/')[-1]}<br>
        Device: {'GPU' if torch.cuda.is_available() else 'CPU'}<br>
        Status: {'Ready' if st.session_state.pipeline else 'Not loaded'}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="header">
    <div class="header-badge">Healthcare AI</div>
    <h1>HealthMind AI</h1>
    <p>Medical reasoning powered by FreedomIntelligence/Apollo-7B · trained on medical-o1-reasoning-SFT</p>
</div>
""", unsafe_allow_html=True)

# ── Example questions ──
st.markdown('<div class="example-label">Quick questions</div>', unsafe_allow_html=True)

example_questions = [
    "Symptoms of Type 2 diabetes?",
    "First-line treatment for hypertension?",
    "What does HIPAA protect?",
    "Social determinants of health?",
    "How is sepsis diagnosed?",
    "What is prior authorization?"
]

cols = st.columns(6)
for i, (col, q) in enumerate(zip(cols, example_questions)):
    with col:
        st.markdown('<div class="example-btn">', unsafe_allow_html=True)
        if st.button(q, key=f"ex_{i}"):
            st.session_state.selected_question = q
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Question input ──
st.markdown('<div class="question-card">', unsafe_allow_html=True)
st.markdown('<div class="question-label">Ask a medical question</div>', unsafe_allow_html=True)

question = st.text_area(
    "Question",
    value=st.session_state.selected_question,
    placeholder="e.g. What are the symptoms of Type 2 diabetes?",
    height=100,
    label_visibility="collapsed"
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    ask = st.button("Ask HealthMind", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        st.session_state.selected_question = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only.
    Do not use for clinical decision making. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Handle query
# ─────────────────────────────────────────────
if ask and question.strip():
    if not st.session_state.pipeline:
        st.error("Please load the model first using the sidebar button.")
    else:
        with st.spinner("HealthMind is thinking..."):
            try:
                start = time.time()
                answer = ask_medical_question(
                    st.session_state.pipeline,
                    st.session_state.tokenizer,
                    question,
                    max_tokens,
                    temperature
                )
                elapsed = round(time.time() - start, 1)

                st.session_state.total_queries += 1
                st.session_state.history.insert(0, {
                    "question": question,
                    "answer": answer,
                    "time": elapsed
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ── Show latest response ──
if st.session_state.history:
    latest = st.session_state.history[0]
    st.markdown(f"""
    <div class="response-card">
        <div class="response-header">
            <span style="font-size:1.3rem;">🏥</span>
            <span class="response-tag">HealthMind Response</span>
            <span style="font-size:0.75rem; color:#8a8070; margin-left:auto;">
                {latest['time']}s
            </span>
        </div>
        <div class="response-text">{latest['answer']}</div>
    </div>
    """, unsafe_allow_html=True)

# ── History ──
if len(st.session_state.history) > 1:
    st.markdown("### Previous Questions")
    for item in st.session_state.history[1:]:
        with st.expander(f"Q: {item['question'][:80]}..."):
            st.write(item['answer'])
