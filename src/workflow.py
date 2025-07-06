import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from models import embeddings, file_path, store_name, persistent_directory, create_vector_store
from prompts import contextualize_q_prompt, qa_prompt

def setup_rag_chain():
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    create_vector_store(docs, embeddings, store_name)

    # Load vector store
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatOpenAI(model="gpt-4o")

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def continual_chat(rag_chain):
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        try:
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            print(f"AI: {result['answer']}")
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=result["answer"]))
        except Exception as e:
            print(f"Error processing query: {e}")