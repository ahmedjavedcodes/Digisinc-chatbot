import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=3
)


loader = PyPDFLoader("Agency.pdf")

pages = loader.load()
# print(len(pages))

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 120,
    separators=["\n\n","\n"," ",""]
)

split_docs = splitter.split_documents(pages)

print(len(split_docs))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents = split_docs,
    embedding = embeddings,
    persist_directory = "./Agency_vectorDb"
)


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': 8,
        'fetch_k': 16, 
        'lambda_mult': 0.4
    }
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


query = "What projects did digisinc cover?"

response = chain.invoke(query)

print(response)
