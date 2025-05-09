from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

def get_custom_retriever(df):
    docs = [Document(page_content=row.to_json()) for _, row in df.iterrows()]
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore.as_retriever()
