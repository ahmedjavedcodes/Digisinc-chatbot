import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import PyPDFLoader
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
    temperature=0.25,
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

custom_separators = [
    "\n# ",   # Agency (h1)
    "\n## ",  # Section (h2)
    "\n### ", # Sub-section (h3)
    "\n#### ",# Topic (h4)
    "\n\n",   # Double Newline (Paragraphs)
    "\n- ",   # Bullets (Dashes)
    "\n* ",   # Bullets (Asterisks)
    "\n",     # Single Newline (Line breaks)
    " ",      # Spaces
    ""        # Characters
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000, 
    chunk_overlap = 200,
    separators = custom_separators
)

split_docs = splitter.split_documents(docs)

print(f"Total chunks created: {len(split_docs)}\n")

for i, doc in enumerate(split_docs):
    # Print the chunk index and the headers the splitter identified
    print(f"--- CHUNK {i} ---")
    print(f"Metadata (Headers): {doc.metadata}")
    
    # Print the first 100 characters to see the content
    content_snippet = doc.page_content.replace('\n', ' ')[:120]
    print(f"Content Snippet: {content_snippet}...")
    print("-" * 20 + "\n")

# create embeddings and store in a vector database

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vector_store = Chroma.from_documents(
#     documents = split_docs,
#     embedding = embeddings,
#     persist_directory = "./agency_vectorDb"
# )

# # retriever to retrieve from vector database 

# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         'k': 12,              # Increase slightly to ensure all service blocks are captured
#         'fetch_k': 50,       # Larger pool for better diversity
#         'lambda_mult': 0.4   # Lower value = MORE diversity
#     }
# )


# # # prompt template 

# template = """
# You are "Digisinc's Strategic AI Envoy." 

# GOAL: 
# Provide a precise and comprehensive response to the user's question using ONLY the provided Context. 

# INSTRUCTIONS:
# 1. **Scope**: If the user asks a general question about services or "what we do," ensure you cover all four core pillars: **Websites & Apps**, **AI Automations & Systems**, **Graphic Design**, and **UI/UX Design**.
# 2. **Specificity**: If the user asks a specific question (e.g., about pricing or a specific project), focus deeply on that data while maintaining the established tone.
# 3. **Tone**: Professional, energetic, and results-oriented. Avoid "fluff" or filler words.
# 4. **Formatting**: Use a clear bulleted list for multiple items. **Bold** key terms, service names, or statistics for scannability.
# 5. **Fallback**: If the specific information requested is not in the context, state: "I'm sorry, I don't have specific details on that. Please contact our team at digisinc.systems@gmail.com".
# 6. **Call to Action**: For any inquiry regarding services or starting a project, conclude by mentioning our contact details: +92 317 8433864 or digisinc.systems@gmail.com.

# Context:
# {context}

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# # # structured output 
# parser = StrOutputParser()

# # # chain 
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | parser
# )


# query = "What services do you provide?"

# response = chain.invoke(query)

# print(response)



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
