# 📄 RAG Document Q&A System using Flask + Gemini + FAISS

A simple **Retrieval-Augmented Generation (RAG)** web application built with **Flask**, **Google Gemini**, **FAISS**, and **HuggingFace Embeddings**.

This project allows users to ask questions based on documents stored in predefined folders.  
It reads files from multiple formats, converts them into text chunks, stores embeddings in a **FAISS vector database**, and answers user queries using **Gemini** with relevant document context.

---

## 🚀 Features

- Ask questions from your local document folders
- Supports multiple file formats:
  - PDF
  - DOCX
  - TXT
  - CSV
  - XLSX
  - PPTX
- Uses **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
- Stores vectors locally using **FAISS**
- Uses **Google Gemini (via LangChain)** for answering questions
- Returns:
  - AI-generated answer
  - Source file names used for retrieval

---

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **LangChain**
- **Google Gemini API**
- **FAISS**
- **HuggingFace Embeddings**
- **PyPDF2**
- **Pandas**
- **python-docx**
- **python-pptx**

---

## 📂 Project Structure

```bash
project/
│
├── demo.py
├── .env
├── faiss_index/              # generated automatically after preload
├── templates/
│   └── demo.html
└── README.md
