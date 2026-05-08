# fast api imports 
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import logging
from fastapi import Header, HTTPException

# langchain imports 
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

# import pinecone 
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone


load_dotenv()

app = FastAPI(title="Digisinc AI Backend")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Restrict CORS to your actual domain
origins = [
    "https://www.digisinc.systems",
    "https://digisinc.systems",
    "http://localhost:3000", # Keep for local testing if needed
]

# 2. Enable CORS so React can communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}

class ChatRequest(BaseModel):
    question: str

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "digisinc-index"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to existing Pinecone index (Ensure you've uploaded data once)
vector_store = Pinecone.from_existing_index(
    index_name="digisinc-index",
    embedding=embeddings,
    text_key="text"
)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': 6, 
        'fetch_k': 10, 
        'lambda_mult': 0.5
        }
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=3
)

template = """
You are "Digisinc's Strategic AI Envoy." 

GOAL: 
Provide a precise and comprehensive response to the user's question using ONLY the provided Context. 

INSTRUCTIONS:
1. **Scope**: If the user asks a general question about services or "what we do," ensure you cover all four core pillars: **Websites & Apps**, **AI Automations & Systems**, **Graphic Design**, and **UI/UX Design**.
2. **Specificity**: If the user asks a specific question (e.g., about pricing or a specific project), focus deeply on that data while maintaining the established tone.
3. **Tone**: Professional, energetic, and results-oriented. Avoid "fluff" or filler words.
4. **Formatting**: Use a clear bulleted list for multiple items. **Bold** key terms, service names, or statistics for scannability.
5. **Fallback**: If the specific information requested is not in the context, state: "I'm sorry, I don't have specific details on that. Please contact our team at digisinc.systems@gmail.com".
6. **Call to Action**: For any inquiry regarding services or starting a project, conclude by mentioning our contact details: +92 317 8433864 or digisinc.systems@gmail.com.

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# # structured output 
parser = StrOutputParser()

# # chain 
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)


# query = "What projects did digisinc cover?"

# response = chain.invoke(query)

# print(response)


# 4. Use Async and Ainvoke for the chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # ainvoke is the asynchronous version of invoke
        response = await chain.ainvoke(request.question)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return {"error": "Something went wrong. Please try again later."}
    


if __name__ == "__main__":
    import uvicorn
    
    # This part ONLY runs if you type 'python app.py' in your terminal.
    # It will NOT run when 'gunicorn' or 'uvicorn' starts the app in the cloud.
    query = "What projects did digisinc cover?"
    try:
        response = chain.invoke(query)
        print(f"✅ Startup Test Successful: {response}")
    except Exception as e:
        print(f"❌ Startup Test Failed: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000)    