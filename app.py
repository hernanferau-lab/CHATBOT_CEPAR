
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

# Configura tu API Key (reemplaza con la tuya)
os.environ["OPENAI_API_KEY"] = "sk-..."

# Configura la página de Streamlit
st.set_page_config(page_title="Asistente CEPAR", layout="wide")
st.title("🤖 Asistente Legal - Demo CEPAR")

# Cargar el documento base (protocolo)
with open("protocolo.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Procesamiento del texto
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_text(raw_text)

# Embeddings y vector store
embedding = OpenAIEmbeddings()
db = FAISS.from_texts(texts, embedding)
retriever = db.as_retriever()

# Modelo de lenguaje
llm = ChatOpenAI(model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Interfaz de usuario
query = st.text_input("Haz tu pregunta sobre protocolos de denuncia:")

if query:
    result = qa_chain.run(query)
    st.success(result)
