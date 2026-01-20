import os
import json
import streamlit as st
from mistralai import Mistral
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List
# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="RAG Chat", layout="wide")

# API KEYS
MISTRAL_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_INDEX_NAME = ""

MODEL_MISTRAL = "mistral-small-latest"
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
top_k_param=10
N_NEIGHBORS = 0
SYSTEM_PROMPT = (
    "Ты высокоуважаемый юрист, который отвечает на вопросы, используя ТОЛЬКО предоставленный контекст "
    "из документа и законы Республики Беларусь. Если в контексте нет информации, так и скажи, "
    "но вначале всегда старайся найти ответ в предоставленном контексте. "
    "На любой вопрос приводи обоснования и цитируй статьи. "
    "Все математические выражения и формулы оформляй, заключая их в символы доллара ($) "
    "для встроенного отображения и двойные символы доллара ($$) для блочного отображения."
)


CHAT_DIR = "chat_sessions"
os.makedirs(CHAT_DIR, exist_ok=True)


@st.cache_resource
def get_mistral_client():
    return Mistral(api_key=MISTRAL_API_KEY)

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

client = get_mistral_client()
index = get_pinecone_index()
embedder = get_embedding_model()

# -------------------------------
# HELPERS
# -------------------------------
def list_chats():
    return [f[:-5] for f in os.listdir(CHAT_DIR) if f.endswith(".json")]

def load_chat(name):
    path = os.path.join(CHAT_DIR, f"{name}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat(name, messages):
    path = os.path.join(CHAT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def first_three_words(text):
    return "-".join(text.strip().split()[:3]).replace("/", "_")


def get_relevant_context(query: str, top_k: int = top_k_param, n_neighbors: int = 1) -> List[str]:
    
    index = get_pinecone_index()
    embedder = get_embedding_model()

    query_embedding = embedder.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=False,
    )

    retrieved_indices = set()
    for match in results['matches']:
        match_id = match['id']  
        
        if "_" in match_id:
            try:
                prefix, num_part = match_id.rsplit('_', 1)
                original_id = int(num_part)

                start_id = max(0, original_id - n_neighbors)
                end_id = original_id + n_neighbors

                for i in range(start_id, end_id + 1):
                    retrieved_indices.add(f"{prefix}_{i}")
            except ValueError:
                retrieved_indices.add(match_id)
        else:
            try:
                original_id = int(match_id)
                for i in range(max(0, original_id - n_neighbors), original_id + n_neighbors + 1):
                    retrieved_indices.add(str(i))
            except ValueError:
                retrieved_indices.add(match_id)

    if not retrieved_indices:
        return []

    fetch_results = index.fetch(ids=list(retrieved_indices))

    final_chunks = []
    
    def sort_key(item):
        id_str = item[0]
        if "_" in id_str:
            try: return int(id_str.rsplit('_', 1)[1])
            except: return 0
        try: return int(id_str)
        except: return 0

    sorted_items = sorted(
        fetch_results['vectors'].items(),
        key=sort_key
    )

    seen_text = set()
    for id_str, vector_data in sorted_items:
        metadata = vector_data.get('metadata', {})
        text = metadata.get('text', '')
        source = metadata.get('source', 'Документ')

        if text and text not in seen_text:
            formatted_chunk = f"[{source}, ID: {id_str}] {text}"
            final_chunks.append(formatted_chunk)
            seen_text.add(text)

    return final_chunks

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Chats")
existing_chats = list_chats()
new_chat_clicked = st.sidebar.button("➕ New Chat")

with st.sidebar.form("chat_select_form"):
    selected_chat = st.radio(
        "Select a chat:",
        ["(none)"] + existing_chats,
        index=0 if new_chat_clicked else None,
    )
    submitted = st.form_submit_button("Open Chat")

# -------------------------------
# STATE HANDLING
# -------------------------------
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "new_chat_mode" not in st.session_state:
    st.session_state.new_chat_mode = False

if new_chat_clicked:
    st.session_state.active_chat = None
    st.session_state.messages = []
    st.session_state.new_chat_mode = True
elif selected_chat and selected_chat != st.session_state.active_chat:
    st.session_state.active_chat = selected_chat
    st.session_state.messages = load_chat(selected_chat)
    st.session_state.new_chat_mode = False

# -------------------------------
# MAIN CHAT WINDOW
# -------------------------------
st.title(" RAG Chat")

if st.session_state.active_chat is None:
    st.info("Select a chat on the left or create a new one.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "retrieved_context" in msg:
                with st.expander("Used relevant context "):
                    for i, ctx in enumerate(msg["retrieved_context"]):
                        st.markdown(f"**Chunk {i+1}:** {ctx}")

# -------------------------------
# USER INPUT
# -------------------------------
user_input = st.chat_input("Введите ваш вопрос...")

if user_input:
    # --- SETUP CHAT NAME ---
    if st.session_state.active_chat is None:
        chat_name = first_three_words(user_input)
        st.session_state.active_chat = chat_name
        st.session_state.messages = []
        st.session_state.new_chat_mode = False
    else:
        chat_name = st.session_state.active_chat

    if chat_name is None:
        st.warning("Please select a chat or click New Chat.")
        st.stop()

    # --- RAG RETRIEVAL ---
    with st.spinner("Searching in the index..."):
        retrieved_chunks = get_relevant_context(user_input, top_k=top_k_param, n_neighbors=N_NEIGHBORS)
    
    context_str = "\n\n---\n\n".join(retrieved_chunks)
    
    augmented_user_input = (
        f"Context from document:\n{context_str}\n\n"
        f"Question: {user_input}"
    )

    # --- UI UPDATES ---
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for m in st.session_state.messages[:-1]: 
        api_messages.append({"role": m["role"], "content": m["content"]})
    
    api_messages.append({"role": "user", "content": augmented_user_input})

    # the answer is streamed gradually
    with st.chat_message("assistant"):
        with st.expander("Relevant chunks", expanded=False):
             for i, chunk in enumerate(retrieved_chunks):
                 st.caption(f"**Chunk {i+1}**")
                 st.text(chunk)

        placeholder = st.empty()
        full_response = ""

        try:
            with client.chat.stream(model=MODEL_MISTRAL, messages=api_messages) as stream:
                for event in stream:
                    data = getattr(event, "data", None)
                    if not data: continue
                    try:
                        delta = data.choices[0].delta.content or ""
                    except:
                        delta = ""
                    if delta:
                        full_response += delta
                        placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error: {e}")

    # --- SAVING ---
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "retrieved_context": retrieved_chunks
    })
    
    save_chat(chat_name, st.session_state.messages)