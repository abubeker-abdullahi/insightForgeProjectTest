from langchain.memory import ConversationBufferMemory

def setup_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
