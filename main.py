import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM

load_dotenv()
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

from huggingface_hub import HfApi
api = HfApi()
print(f"Logged in as: {api.whoami()['name']}")


# llm 
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=3,
)
# splitting the heading and loading the document 
file = "Agency_Profile.md"
with open(file, "r", encoding="utf-8") as f:
    markdown_content = f.read()

headers_to_split_on = [
    ("#", "Agency_Name"),
    ("##", "Section"),
    ("###", "Sub_Section"),
    ("####", "Topic"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on= headers_to_split_on,
    strip_headers = False
)

docs = markdown_splitter.split_text(markdown_content)

# splitting the loaded document 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 100,
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n|", "\n"]
)

split_docs = splitter.split_documents(docs)

# create embeddings and store in a vector database

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents = split_docs,
    embedding = embeddings,
    persist_directory = "./agency_vectorDb"
)

# retriever to retrieve from vector database 

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs={'k': 8}
)


# prompt template 

template = """
You are "Digisinc's Strategic AI Envoy" a high-performance, results-driven professional assistant for Digisinc Marketing Agency.
GOAL: 
Provide helpful information about Digisinc's services, pricing, and statistics based ONLY on the provided context.
RULES OF ENGAGEMENT:
1. ONLY use the provided Context. If the information is not there, say: "I'm sorry, I don't have specific details on that. Please contact our team at digisinc.systems@gmail.com."
2. TONE: Professional, energetic, and concise. Avoid "fluff."
3. FORMATTING: Use bullet points and bold text for pricing or statistics to make them easy to read.
4. CALL TO ACTION: If the user asks about starting a project, mention they can reach out via +92 317 8433864 or "digisinc.systems@gmail.com".

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# structured output 
parser = StrOutputParser()

# chain 
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)


query = "What services do you provide?"

response = chain.invoke(query)

print(response)



# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are the helpful assistant"),
#     ("human", "{user_input}")
# ]
# )

# parser = StrOutputParser()

# input = "What is token in the Gen Ai?"

# chain = prompt | llm | parser

# response = chain.invoke({"user_input":input})


# print(response)
