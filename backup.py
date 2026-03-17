from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Summarizer pipeline initialization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

app = Flask(__name__)

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    all_text_chunks = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        text_chunks = get_text_chunks(text)
        all_text_chunks.extend([{"text": chunk, "metadata": {"file_name": pdf.filename}} for chunk in text_chunks])
    return all_text_chunks

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Vector store creation
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk['text'] for chunk in text_chunks]
    metadatas = [chunk['metadata'] for chunk in text_chunks]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")

# Chain for question-answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "Answer is not available in the context."\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    file_names = [doc.metadata.get('file_name', 'Unknown file') for doc in docs]
    unique_file_names = list(set(file_names))

    sources = ", ".join(unique_file_names) 
    return response["output_text"], sources

@app.route('/')
def index():
    return render_template('backup.html')

@app.route('/upload', methods=['POST'])
def upload():
    start_time = time.time()  # Start timing
    pdf_docs = request.files.getlist('pdf_files')
    if pdf_docs:
        text_chunks = get_pdf_text(pdf_docs)
        get_vector_store(text_chunks)
        end_time = time.time()  # End timing
        duration = end_time - start_time
        print(f"Upload and processing took {duration:.2f} seconds.")  # Log duration
        return jsonify({"message": "Documents processed successfully!"})
    return jsonify({"error": "Please upload PDF files."})

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question')
    if user_question:
        answer, sources = user_input(user_question)  
        return jsonify({"answer": answer, "sources": sources})
    return jsonify({"error": "Please provide a question."})

if __name__ == "__main__":
    app.run(debug=True)
