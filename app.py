import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from cohere import Client as CohereClient
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
from dotenv import load_dotenv
import numpy as np
import re



load_dotenv()

COLLECTION_NAME = "VerilogTest1"
VECTOR_SIZE = 1536
SIMILARITY_THRESHOLD = 0.78
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

qdrant_client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
cohere_client = CohereClient(COHERE_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chatbot.log")]
)
logger = logging.getLogger(__name__)



def looks_like_verilog(text: str) -> bool:
    text = text.strip()
    if len(text) < 20:
        return False
    lower = text.lower()
    return any(kw in lower for kw in [
        'module', 'endmodule', 'input', 'output', 'inout', 'wire', 'reg',
        'always', 'assign', 'initial', 'posedge', 'negedge', '@(', 'begin', 'end'
    ])

def simple_verilog_check(code: str) -> list[str]:
    issues = []
    lines = code.splitlines()

    # Global balances
    paren_count = code.count('(') - code.count(')')
    if paren_count != 0:
        issues.append(f"Unbalanced parentheses (difference: {paren_count:+d}) – check opening/closing ()")

    bracket_count = code.count('[') - code.count(']')
    if bracket_count != 0:
        issues.append(f"Unbalanced brackets (difference: {bracket_count:+d}) – check bus widths like [n:0]")

    begin_end_count = code.lower().count('begin') - code.lower().count('end')
    if begin_end_count != 0:
        issues.append(f"Unbalanced begin/end (difference: {begin_end_count:+d}) – ensure every begin has an end")

    module_endmodule = 'module' in code.lower() and 'endmodule' not in code.lower()
    if module_endmodule:
        issues.append("Module declaration without endmodule – add 'endmodule;' at the end")

    # Line-by-line checks
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('`'):
            continue

        # Suspicious always
        if 'always' in stripped.lower() and '@' not in stripped:
            issues.append(f"Line {i}: 'always' block without sensitivity list (@*) or (@(posedge clk)) – may cause simulation issues")

        # Comment placement
        if ';' in stripped and '//' in stripped and stripped.find('//') > stripped.find(';'):
            issues.append(f"Line {i}: Comment after semicolon – consider moving comment to a separate line or before ;")

        # Possible missing ;
        if i < len(lines):
            next_l = lines[i].strip()
            if (stripped and not stripped.endswith(';') and not stripped.endswith('{')
                and not stripped.endswith(')') and not stripped.startswith('end')
                and next_l and not next_l.startswith('end') and not next_l.startswith('//')):
                issues.append(f"Line {i}: Possible missing semicolon at end of statement")

        # Common syntax: input/output without type
        if any(kw in stripped.lower() for kw in ['input', 'output', 'inout']) and '[' in stripped and 'wire' not in stripped.lower() and 'reg' not in stripped.lower():
            issues.append(f"Line {i}: Port declaration may need type (e.g., input wire [7:0] data)")

    return issues[:10]  # Limit to avoid overwhelming



@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120), retry=retry_if_exception_type(Exception))
def get_embedding(text):
    resp = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120), retry=retry_if_exception_type(Exception))
def search_qdrant(query_embedding, limit=200):
    if query_embedding is None:
        return []
    try:
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True
        ).points
        return [(hit.payload.get("code"), hit.payload.get("title"), hit.vector) for hit in search_result]
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        return []

def cosine_similarity(a, b):
    if a is None or b is None: return 0.0
    a, b = np.array(a), np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def rerank_results(query, chunks):
    try:
        docs = [c[0] for c in chunks if c[0]]
        if not docs: return []
        resp = cohere_client.rerank(query=query, documents=docs, top_n=min(30, len(docs)))
        results = getattr(resp, "results", [])
        if not results:
            return [{"code": c[0], "title": c[1], "relevance_score": 0} for c in chunks[:50]]
        idx_to_chunk = {i: c for i, c in enumerate(chunks)}
        return [
            {"code": idx_to_chunk[item.index][0],
             "title": idx_to_chunk[item.index][1],
             "relevance_score": item.relevance_score}
            for item in results
        ]
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        return [{"code": c[0], "title": c[1], "relevance_score": 0} for c in chunks[:50]]

def generate_answer(messages_for_llm, context=None, detected_issues=None):
    ctx_text = ""
    if context:
        ctx_text = "\n".join([
            f"Source: {r['title']}\nCode: {r['code'][:1100]}{'...' if len(r['code'])>1100 else ''}"
            for r in context[:10]
        ])

    issues_text = ""
    if detected_issues:
        issues_text = "\nDetected issues in user-provided code:\n" + "\n".join([f"- {issue}" for issue in detected_issues]) + "\nPlease address and correct these in your response."

    system = f"""You are a Verilog / RTL / VLSI expert.
• Generate VLSI/Verilog code for requests.
• If user pastes code: analyze it, comment on errors/improvements, and provide corrected version if issues found.
• Use context when helpful.
• Show clean code with minimal comments. Start code blocks with: **Here is the Verilog code for [topic]:**
• Be concise and technical.
{issues_text}

Context (if relevant):
{ctx_text}"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}] + messages_for_llm,
            max_tokens=2200,
            temperature=0.65
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Error generating response."

def hybrid_search(query):
    emb = get_embedding(query)
    if not emb: return []
    results = search_qdrant(emb)
    if not results: return []
    sim = cosine_similarity(emb, results[0][2])
    if sim >= SIMILARITY_THRESHOLD:
        return rerank_results(query, results)
    return []



st.set_page_config(page_title="Silicore-X", page_icon="⚡", layout="wide")

# Better looking chat bubbles
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem 1.2rem;
        border-radius: 0
        margin: 0.5rem 0;
    }
    .user .stChatMessage {
        background: #0d47a1;
        color: white;
    }
    .assistant .stChatMessage {
        background: #1e1e2f;
        color: #e0e0ff;
    }
    .stChatInput > div:first-child {
        padding-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Silicore-X⚡Verilog🤖")
st.markdown("""
Ask your questions related to **Verilog, VLSI design**, **IC architecture**, **digital & analog circuits**, **logic synthesis**, **layout optimization**, and more ⚡💡

The **Silicore-X agent**  will use **hybrid mode** ⚙️🚀
""")
# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show conversation — oldest → newest (top to bottom)
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    avatar = "👤" if role == "user" else "⚡"

    with st.chat_message(role, avatar=avatar):
        if role == "user" and looks_like_verilog(content):
            st.code(content, language="verilog")
        else:
            st.markdown(content)

# ── Input box at bottom ──
if user_input := st.chat_input("Ask Verilog related question or paste code"):
    # Add & show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        if looks_like_verilog(user_input):
            st.code(user_input, language="verilog")
        else:
            st.markdown(user_input)

    # Assistant response area
    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Thinking..."):
            issues = []
            if looks_like_verilog(user_input):
                issues = simple_verilog_check(user_input)
                if issues:
                    st.warning("**Possible issues detecting in your code:**")
                    for issue in issues:
                        st.write("• " + issue)
                    st.info("Generating updated code with corrections...")

            context = hybrid_search(user_input)

            # Last 10 full turns (≈20 messages)
            recent = st.session_state.messages[-20:]
            llm_msgs = [{"role": m["role"], "content": m["content"]} for m in recent]

            answer = generate_answer(llm_msgs, context, detected_issues=issues)

            st.markdown(answer)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Force scroll to bottom
    st.markdown(
        """
        <script>
            const elem = window.parent.document.querySelector('.stApp > section > div');
            if (elem) elem.scrollTop = elem.scrollHeight;
        </script>
        """,
        unsafe_allow_html=True
    )
