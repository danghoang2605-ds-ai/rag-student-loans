import os
from dotenv import load_dotenv
from config import CUSTOM_PROMPT_TEMPLATE
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# load_dotenv(override=False)

source_files = [
    "data/uga_resources_all.json",
    "data/uga_osfa_faqs.json",
    "data/cfpb_docs.json",
]
all_docs = []
for filepath in source_files:
    with open(filepath, encoding="utf-8") as f:
        all_docs.extend(json.load(f))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

documents = []
for doc in all_docs:
    q = doc.get("question", "")
    a = doc.get("answer", "")
    text = f"Question: {q}\nAnswer: {a}" if q else a

    chunks = text_splitter.split_text(text)
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

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

INDEX_DIR = "indexes/faiss_index_local"

if os.path.isdir(INDEX_DIR):
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    print(f"No local index found at {INDEX_DIR}, building a new one from {len(documents)} chunks...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved new FAISS index to {INDEX_DIR}")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_retrieve(query):
    base_docs = vectorstore.similarity_search(query, k=25)
    pairs = [(query, d.page_content) for d in base_docs]
    scores = cross_encoder.predict(pairs)
    for d, s in zip(base_docs, scores):
        d.metadata["rerank_score"] = float(s)
    return sorted(base_docs, key=lambda d: d.metadata["rerank_score"], reverse=True)[:4]


class RerankRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return rerank_retrieve(query)


retriever = RerankRetriever()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3, max_tokens=400)

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# LCEL chain — replaces the deleted RetrievalQA.from_chain_type
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | CUSTOM_PROMPT
    | llm
    | StrOutputParser()
)


def ask(question: str):
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return {"result": answer, "source_documents": docs}


if __name__ == "__main__":
    question = "How do I pay off my student loans I dont have the money right now"

    try:
        result = ask(question)
    except Exception as e:
        print(f"\n[ERROR] Something went wrong while answering the question: {e}")
        raise

    print("\nQUESTION:", question)
    print("\nANSWER:", result["result"])
    print("\n--- Retrieved Chunks ---")
    for i, d in enumerate(result["source_documents"]):
        print(f"[{i}] Source: {d.metadata['source']}")
        print(d.page_content[:500], "\n")