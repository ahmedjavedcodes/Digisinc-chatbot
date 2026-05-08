import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Load and Split PDF
loader = PyPDFLoader("Agency.pdf")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
docs = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# 3. Connect to Pinecone directly
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("digisinc-index")

print(f"🔄 Processing {len(docs)} chunks...")

vectors = []
for i, doc in enumerate(docs):
    # This turns the text into the 1024-dimension numbers
    vector_values = embeddings.embed_query(doc.page_content)
    
    vectors.append({
        "id": f"chunk_{i}",
        "values": vector_values,
        "metadata": {
            "text": doc.page_content,
            "page": doc.metadata.get("page", 0)
        }
    })

index.upsert(vectors=vectors)

print(f"✅ Success! Your PDF is now searchable in the Pinecone cloud.")