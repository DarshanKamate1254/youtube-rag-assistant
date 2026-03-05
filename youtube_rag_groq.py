import streamlit as st
import os
import tempfile
import whisper
import tiktoken
import pandas as pd
import yt_dlp

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🎥 YouTube RAG AI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM STYLING
# ==========================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    font-size: 18px;
    color: #9CA3AF;
    margin-bottom: 30px;
}
.stButton>button {
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}
.stTextArea textarea {
    border-radius: 10px;
}
.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD ENV
# ==========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("⚙️ Configuration")

    MAX_TOKENS = st.slider("Max Tokens Per Chunk", 200, 1500, 500)
    whisper_model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium"],
        index=1
    )

    st.markdown("---")
    st.subheader("🔑 API Status")

    if GROQ_API_KEY:
        st.success("Groq Connected")
    else:
        st.error("Groq Key Missing")

    if PINECONE_API_KEY:
        st.success("Pinecone Connected")
    else:
        st.error("Pinecone Key Missing")

# ==========================
# CONFIG
# ==========================
INDEX_NAME = "another-tube"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
tokenizer = tiktoken.get_encoding("cl100k_base")

# ==========================
# FUNCTIONS (UNCHANGED)
# ==========================
def download_audio(youtube_url, output_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "audio.%(ext)s"),
        "quiet": True,
        "restrictfilenames": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get("title", "YouTube Video")

    return os.path.join(output_path, "audio.mp3"), title


def transcribe(youtube_url, model):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path, title = download_audio(youtube_url, tmpdir)
        result = model.transcribe(audio_path, fp16=False)
    return title, youtube_url, result["text"].strip()


def split_into_many(text, max_tokens):
    sentences = text.split(". ")
    n_tokens = [len(tokenizer.encode(" " + s)) for s in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


# ==========================
# MAIN HEADER
# ==========================
st.markdown('<div class="main-title">🎥 YouTube RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask intelligent questions from YouTube videos using AI</div>', unsafe_allow_html=True)

# ==========================
# INPUT SECTION
# ==========================
col1, col2 = st.columns([3, 1])

with col1:
    youtube_links = st.text_area(
        "Paste YouTube URLs (one per line)",
        height=150
    )

with col2:
    st.info("🚀 Tips:\n\n• One URL per line\n• Works best for lectures\n• Avoid very long videos")

# ==========================
# PROCESS BUTTON
# ==========================
if st.button("🚀 Process Videos", use_container_width=True):

    urls = [u.strip() for u in youtube_links.split("\n") if u.strip()]

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Loading Whisper model..."):
        whisper_model = whisper.load_model(whisper_model_size)

    transcriptions = []

    for i, url in enumerate(urls):
        status_text.text(f"Transcribing: {url}")
        transcriptions.append(transcribe(url, whisper_model))
        progress_bar.progress((i + 1) / len(urls))

    df = pd.DataFrame(transcriptions, columns=["title", "url", "text"])
    df["tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    data = []
    for _, row in df.iterrows():
        if row["tokens"] <= MAX_TOKENS:
            data.append((row["title"], row["url"], row["text"]))
        else:
            for chunk in split_into_many(row["text"], MAX_TOKENS):
                data.append((row["title"], row["url"], chunk))

    chunked_df = pd.DataFrame(data, columns=["title", "url", "text"])
    chunked_df.to_csv("video_text.csv", index=False)

    # Metrics
    colA, colB, colC = st.columns(3)
    colA.metric("Videos Processed", len(urls))
    colB.metric("Chunks Created", len(chunked_df))
    colC.metric("Embedding Model", "MiniLM")

    # VECTOR STORE
    loader = CSVLoader(file_path="video_text.csv")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

    vectorstore.add_documents(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Provide a clear and structured answer.
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    st.session_state.rag_chain = rag_chain
    st.success("RAG System Ready 🎯")


# ==========================
# CHAT SECTION (Modern)
# ==========================
if "rag_chain" in st.session_state:

    st.markdown("---")
    st.subheader("💬 Chat With Your Videos")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask something about the video...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])