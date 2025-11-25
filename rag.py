import os
from dotenv import load_dotenv
from config import CUSTOM_PROMPT_TEMPLATE
import json
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from sentence_transformers import CrossEncoder
from langchain.prompts import PromptTemplate

import numpy as np


load_dotenv()

#Load specific documents from data folder
source_files = [
    "data/uga_resources_all.json",
    "data/uga_osfa_faqs.json",
    "data/cfpb_docs.json",
]
all_docs = []
for filepath in source_files:
    with open(filepath) as f:
        all_docs.extend(json.load(f))

# Chunk docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      
    chunk_overlap=50,    
    separators=["\n\n", "\n", ". ", " ", ""]  
)

#Turn docs into langchain documents
documents = []
for doc in all_docs:
    q = doc.get("question", "")
    a = doc.get("answer", "")

    if q:
        text = f"Question: {q}\nAnswer: {a}"
    else:   
        text = a

    chunks = text_splitter.split_text(text)
    # Add source label
    for chunk in chunks:
        if len(chunk) < 80:
            continue
        source_url = doc.get("url", "unknown")
        if "uga.edu" in source_url.lower():
            label = "UGA FAQ"
        elif any(x in source_url.lower() for x in ["consumerfinance.gov", "studentaid.gov", "ed.gov"]):
            label = "CFPB/DOE"
        else:
            label = "Other Source"

        documents.append(Document(
            page_content=f"[{label}] {chunk}",
            metadata={"source": source_url, "source_type": label}
        ))

        
# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Build FAISS vectorstore
INDEX_DIR = "indexes/faiss_index"
if os.path.isdir(INDEX_DIR):
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)

# Retriever with reranking algorithm
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_retrieve(query):
    base_docs = base_retriever.vectorstore.similarity_search(query, k=25)
    pairs = [(query, d.page_content) for d in base_docs]
    scores = cross_encoder.predict(pairs)
    for d, s in zip(base_docs, scores):
        d.metadata["rerank_score"] = float(s)
    return sorted(base_docs, key=lambda d: d.metadata["rerank_score"], reverse=True)[:4]

# Custom reranker with proper typing and methods
class RerankRetriever(BaseRetriever):
    def __init__(self):
        super().__init__()

    def _get_relevant_documents(self, query):
        return rerank_retrieve(query)
    
retriever = RerankRetriever()

# Claude 3 haiku LLM
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3, max_tokens=400)


CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Chain to inject relevant documents into LLM prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",   
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": CUSTOM_PROMPT,
        "document_separator": "\n\n"
    },

)

if __name__ == "__main__":
    question = "How do I pay off my student loans I dont have the money right now"
    result = qa_chain.invoke({"query": question})
    
    print("\nQUESTION:", question)
    print("\nANSWER:", result["result"])
    print("\n--- Retrieved Chunks ---")
    for i, d in enumerate(result["source_documents"]):
        print(f"[{i}] Source: {d.metadata['source']}")
        print(d.page_content[:500], "\n")