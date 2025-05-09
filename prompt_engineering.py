from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def generate_response(query, retriever, memory):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, memory=memory)
    return qa_chain.run(query)
