# PDF-General-Q-A-Chatbot-Streamlit-Gemini-FAISS-LangChain-
This project is an AI-powered chatbot that can answer:

Questions from uploaded PDF documents

General questions asked by the user

Questions using chat history (memory)

It uses Google Gemini, FAISS vector search, and LangChain to deliver accurate, context-aware answers in real time.

🚀 Features
🔍 PDF Question Answering

Upload any PDF

Extracts text using PyPDF2

Splits text into chunks with RecursiveCharacterTextSplitter

🤖 AI-Powered Responses

Uses Gemini 1.5 Flash (or Pro) to generate detailed answers

Context-aware: uses previous Q&A in the prompt

⚡ Vector Search with FAISS

Embeds text using MiniLM (HuggingFace Embeddings)

Retrieves most relevant chunks for each query

🎨 Interactive Streamlit UI

Sidebar PDF upload section

Sidebar chat history

Clean input/output display

🧠 How It Works

User uploads a PDF

PDF text is extracted

Text is split into overlapping chunks

Each chunk is converted into vector embeddings

FAISS stores embeddings for similarity search

User asks a question

Related chunks are retrieved

Gemini generates the answer using:

Retrieved PDF context

Chat history

User question

📦 Tech Stack
Component	Technology
LLM
Google Gemini API
Framework	
Streamlit
Embeddings	
HuggingFace MiniLM
Vector Store
FAISS
Text Processing	
LangChain
PDF Reader
PyPDF2
