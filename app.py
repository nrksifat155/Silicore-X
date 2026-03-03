import streamlit as st
from qdrant_client import QdrantClient
import google.generativeai as genai
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv
import numpy as np
from google.api_core import exceptions

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION (Loading from .env)
# ─────────────────────────────────────────────────────────────

load_dotenv()

# .env থেকে API কি-গুলো লিস্ট আকারে নেওয়া হচ্ছে
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5")
]

# ফিল্টার আউট None values (যদি ৫টির কম কী দেওয়া থাকে)
API_KEYS = [k for k in API_KEYS if k]

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")

GENERATIVE_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL  = "models/gemini-embedding-001" 

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ─────────────────────────────────────────────────────────────
# 2. ANALYSIS & ERROR CHECKING LOGIC
# ─────────────────────────────────────────────────────────────

def looks_like_verilog(text: str) -> bool:
    text = text.strip()
    if len(text) < 20: return False
    keywords = ['module', 'endmodule', 'input', 'output', 'wire', 'reg', 'always', 'assign', 'posedge', 'negedge']
    return any(kw in text.lower() for kw in keywords)

def perform_internal_check(code: str) -> dict:
    report = {"Syntax": [], "Conceptual": [], "Mathematical/Logical": []}
    if code.count('(') != code.count(')'): report["Syntax"].append("Unbalanced parentheses ()")
    if code.count('[') != code.count(']'): report["Syntax"].append("Unbalanced brackets []")
    if code.lower().count('begin') != code.lower().count('end'): report["Syntax"].append("Unbalanced begin/end")
    if 'module' in code.lower() and 'endmodule' not in code.lower(): report["Syntax"].append("Missing endmodule")
    
    lines = code.splitlines()
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if 'always' in s.lower() and '@' not in s:
            report["Conceptual"].append(f"Line {i}: always block missing sensitivity list")
    return report

# ─────────────────────────────────────────────────────────────
# 3. CORE GENERATION LOGIC (With API Key Switching)
# ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20))
def get_embedding(text: str):
    if not API_KEYS: return None
    genai.configure(api_key=API_KEYS[0])
    return genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_query")["embedding"]

def hybrid_search(query: str):
    try:
        emb = get_embedding(query)
        hits = qdrant_client.query_points(collection_name="VerilogTest1", query=emb, limit=5, with_payload=True).points
        return [{"code": h.payload.get("code"), "title": h.payload.get("title")} for h in hits]
    except Exception: return []

def generate_answer(messages, context, error_report, has_errors, is_verilog_code):
    ctx_text = "\n".join([f"Source: {r['title']}\nCode: {r['code'][:800]}" for r in context])
    
    system_prompt = f"""
You are a Verilog/VLSI Expert. Your responses must be sharp, concise, and technically accurate.

### STRICT RULES:
1. **If the code is correct ✅:** - Start exactly with: "The code is correct ✅"
   - Do NOT provide a truth table or a rewritten code block.
   - Provide only 3-4 brief, high-level recommendations for optimization or best practices in bullet points.

2. **If the code is incorrect ❌:**
   - Start exactly with: "The code is incorrect ❌"
   - Identify the **Error Type** (e.g., Syntax, Conceptual, or Logical).
   - Provide the **Corrected Code** in a single block starting with: **Here is the Verilog code for [topic]:**
   - Do NOT provide a truth table or unnecessary explanations.

3. **General Generation (No code provided by user):**
   - Provide the Verilog code.
   - Provide a Markdown **Truth Table** and mention the **Logic Gates** used.
4. Answers must be concise and clear.

Context from Database:
{ctx_text}
"""

    for key in API_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GENERATIVE_MODEL)
            response = model.generate_content(f"{system_prompt}\n\nUser: {messages[-1]['content']}")
            return response.text.strip()
        except exceptions.ResourceExhausted:
            continue
        except Exception:
            continue
    
    return "All API key limits are exhausted. Please try again later."

# ─────────────────────────────────────────────────────────────
# 4. STREAMLIT UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Silicore-X", page_icon="⚡", layout="wide")
st.title("Silicore-X⚡")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])

if user_input := st.chat_input("Generate a circuit or paste code to analyze..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="👤"):
        is_verilog_input = looks_like_verilog(user_input)
        if is_verilog_input: 
            st.code(user_input, language="verilog")
        else: 
            st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚡"):
        status_label = "Analyzing..." if is_verilog_input else "Generating..."
        with st.spinner(status_label):
            error_report = perform_internal_check(user_input) if is_verilog_input else {"Syntax":[], "Conceptual":[], "Mathematical":[]}
            has_errors = any(len(v) > 0 for v in error_report.values())
            context = hybrid_search(user_input)
            answer = generate_answer(st.session_state.messages, context, error_report, has_errors, is_verilog_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
