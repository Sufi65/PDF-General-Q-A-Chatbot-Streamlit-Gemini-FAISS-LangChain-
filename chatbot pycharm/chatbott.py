import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# --- Gemini API Key ---
GEMINI_API_KEY = "AIzaSyCdlM8s6RUN54g5QlSXB76HxoAQjXTtJvk"
genai.configure(api_key=GEMINI_API_KEY)

st.header("📄 PDF & General Q&A Chatbot ! ")

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar container ---
with st.sidebar:
    st.title("Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    st.markdown("### Chat History")  # header only
    questions_placeholder = st.empty()  # container for live questions only

# --- Initialize FAISS vector store ---
vector_store = None
retriever = None

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- User input ---
user_question = st.text_input("Ask a question:")

if user_question:
    model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-1.5-pro

    if retriever is not None:
        docs = retriever.get_relevant_documents(user_question)
        context = "\n".join([doc.page_content for doc in docs])
        previous_qa = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        prompt = f"{previous_qa}\n\nUse the following context to answer the new question.\n\nContext:\n{context}\n\nQuestion: {user_question}\nAnswer:"
    else:
        previous_qa = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        prompt = f"{previous_qa}\n\nAnswer the following question:\n\n{user_question}\nAnswer:"

    response = model.generate_content(prompt)
    answer_text = response.text

    # --- Save Q&A ---
    st.session_state.chat_history.append((user_question, answer_text))

# --- Update sidebar with questions only ---
questions_html = '<div style="overflow-y:auto; max-height:500px;">'
for i, (q, _) in enumerate(st.session_state.chat_history):
    q_html = q.replace("\n", "<br>")
    questions_html += f"<b>Q{i+1}:</b> {q_html}<br><hr>"
questions_html += "</div>"
questions_placeholder.markdown(questions_html, unsafe_allow_html=True)

# --- Show current answer in main page ---
if user_question:
    st.markdown("### Answer:")
    st.write(answer_text)
