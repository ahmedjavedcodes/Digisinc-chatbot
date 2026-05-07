import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Load your local PDF
loader = PyPDFLoader("Agency.pdf")
data = loader.load()

# 2. Split into chunks so the AI can read small pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# 3. Setup Embeddings (Must match your Pinecone index dimensions: 384)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name="digisinc-index"
)

print("✅ Success! Your PDF content is now safely stored in the Pinecone cloud.")