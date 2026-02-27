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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chatbot.log")]
)
logger = logging.getLogger(__name__)


# API keys and configuration
COLLECTION_NAME = "VerilogTest1"
VECTOR_SIZE = 1536
SIMILARITY_THRESHOLD = 0.78
# API keys and configuration
COLLECTION_NAME = "VerilogTest1"
VECTOR_SIZE = 1536  # Matches text-embedding-ada-002
EMBEDDING_MODEL = "text-embedding-ada-002"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
cohere_client = CohereClient(COHERE_API_KEY)

# ---------------- Helpers ----------------
def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

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
        logger.error(f"Error searching Qdrant: {e}")
        return []

def rerank_results(query, chunks):
    try:
        documents = [chunk[0] for chunk in chunks if chunk[0] is not None]
        if not documents:
            return []
        response = cohere_client.rerank(query=query, documents=documents, top_n=min(30, len(documents)))
        results = getattr(response, "results", None)
        if not results:
            return [{"code": c[0], "title": c[1], "relevance_score": 0} for c in chunks[:50]]
        index_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
        reranked = []
        for item in results:
            idx = item.index
            score = item.relevance_score
            chunk_code, title, _ = index_to_chunk[idx]
            reranked.append({"code": chunk_code, "title": title, "relevance_score": score})
        return reranked
    except Exception as e:
        logger.error(f"Error reranking with Cohere: {e}")
        return [{"code": c[0], "title": c[1], "relevance_score": 0} for c in chunks[:50]]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120), retry=retry_if_exception_type(Exception))
def generate_answer(query, context=None):
    try:
        context_text = ""
        if context:
            context_text = "\n".join([f"Source: {r['title']}\nCode snippet: {r['code'][:1000]}..." for r in context[:15]])
        prompt = (
            f"Context: {context_text}\n\n"
            f"User input: {query}\n\n"
            f"• You are a VLSI engineer specialized in Verilog/RTL code.\n"
            f"• If context is given, provide working Verilog snippets from it.\n"
            f"• If no context or low similarity, generate code from your own knowledge.\n"
            f"• Include minimal comments; cite sources if available.\n"
            f"• If input code is correct, mention 'No errors detected.'\n"
            f"• Keep responses concise, technical, and practical."
        )
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Unable to generate answer."

def hybrid_search(query):
    query_emb = get_embedding(query)
    if query_emb is None:
        return generate_answer(query)
    
    db_results = search_qdrant(query_emb)
    if db_results:
        top_code, top_title, top_vector = db_results[0]
        sim = cosine_similarity(query_emb, top_vector)
        if sim >= SIMILARITY_THRESHOLD:
            logger.info(f"DB match found with similarity {sim:.2f}, using DB snippet")
            reranked = rerank_results(query, db_results)
            return generate_answer(query, context=reranked)
        else:
            logger.info(f"No high similarity DB match ({sim:.2f}), using GPT own knowledge")
    else:
        logger.info("No DB results, using GPT own knowledge")
    return generate_answer(query)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VLSI Silicore-X", page_icon="🤖")
st.title("Silicore-X 🤖💡🔬")
st.markdown("""
Ask your questions related to **Verilog, VLSI design**, **IC architecture**, **digital & analog circuits**, **logic synthesis**, **layout optimization**, and more 🛠️📐.

The **Silicore-X agent** will use **hybrid mode**
""")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_area("Enter your question or paste Verilog code:")
if st.button("Generate Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question or Verilog code.")
    else:
        with st.spinner("Generating answer... ⏳"):
            answer = hybrid_search(user_input)
        st.session_state.conversation.append({"user": user_input, "bot": answer})

# Display conversation
for turn in st.session_state.conversation:
    st.markdown("**User Input:**")
    st.code(turn["user"], language="verilog")
    st.markdown("**Silicore-X Feedback:**")
    st.markdown(turn["bot"])