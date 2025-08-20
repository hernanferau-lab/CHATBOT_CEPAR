
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verifica que la clave esté presente
if not openai_api_key:
    st.error("🔐 No se ha configurado la clave OPENAI_API_KEY. Agrega tu clave en Secrets de Streamlit Cloud.")
    st.stop()

# Título de la App
st.title("🤖 Asistente Legal - Demo CEPAR")

# Carga y embebe el documento
loader = TextLoader("protocolo.txt")
documents = loader.load()
texts = [doc.page_content for doc in documents]
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_texts(texts, embedding)

# Inicializa el modelo
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=retriever
)

# UI de interacción
pregunta = st.text_input("¿En qué puedo ayudarte hoy?")
if pregunta:
    respuesta = qa_chain.run(pregunta)
    st.write("Respuesta:", respuesta)
