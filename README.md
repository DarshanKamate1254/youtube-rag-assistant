# 🎥 YouTube RAG Assistant  

Ask intelligent questions from YouTube videos using AI.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that transcribes YouTube videos, converts transcripts into embeddings, stores them in a vector database, and enables context-aware question answering using a Large Language Model.

> ⚠️ This project was developed as a learning-based reimplementation inspired by a Kaggle notebook (reference provided below).  
> The architecture was independently rebuilt, modularized, and extended for better structure and production-style design.

---

## 🚀 Features  

- 🎧 Download and transcribe YouTube videos using OpenAI Whisper  
- ✂️ Smart token-based transcript chunking  
- 🧠 Embeddings using HuggingFace MiniLM  
- 🗂 Vector storage with Pinecone  
- ⚡ Fast LLM inference via Groq (LLaMA 3)  
- 💬 Chat-style Q&A interface using Streamlit  
- 🔎 Context-aware answers using RAG pipeline  
- 🧱 Modular, scalable project structure  

---

## 🧠 Learning Background & Reference  

This project was inspired by the following Kaggle notebook:

🔗 https://www.kaggle.com/code/derrickmwiti/langchain  

I referred to the overall RAG workflow and LangChain integration concepts from this notebook. However:

- The implementation was rewritten independently  
- The pipeline was adapted for YouTube-based retrieval  
- Integrated Groq LLM for faster inference  
- Added token-aware chunking  
- Structured as a modular Streamlit application  
- Improved readability and production-style organization  

This repository represents my hands-on learning and independent implementation of RAG systems.

---

## 🏗 Tech Stack  

- Python 3.12  
- Streamlit  
- Whisper  
- Pinecone  
- LangChain  
- Groq LLM  
- HuggingFace Embeddings  
- yt-dlp  
- tiktoken  

---
<p align="center">
  <img src="yt_rag.png">
</p>

<p align="center">
  <img src="yt_rag2.png">
</p>

## 📦 Installation  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/YOUR_USERNAME/youtube-rag-assistant.git
cd youtube-rag-assistant