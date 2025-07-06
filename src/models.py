import os
from glob import glob
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "assets", "books")
store_name = "chroma_db_with_metadata"
persistent_directory = os.path.join(current_dir, "db", store_name)
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(db_dir):
    os.makedirs(db_dir)

book_files = glob(os.path.join(books_dir, "*.txt"))
if not book_files:
    raise FileNotFoundError(f"No text files found in {books_dir}. Please check the path.")

all_docs = []
for file_path in book_files:
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    all_docs.extend(docs)

def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(current_dir, "db", store_name)
    if not os.path.exists(persistent_directory):
        print(f"\nCreating vector store {store_name}")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"Finished creating vector store {store_name}")
        return True
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")
        return False

def query_vector_store(store_name, query, embedding_function, search_type="similarity", k=3, score_threshold=None):
    persistent_directory = os.path.join(current_dir, "db", store_name)
    if os.path.exists(persistent_directory):
        print(f"\nQuerying the Vector Store {store_name}")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        search_kwargs = {"k": k}
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        try:
            relevant_docs = retriever.invoke(query)
            print(f"\nRelevant Documents for {store_name}")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"Document {i}:\n{doc.page_content}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
            return relevant_docs
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []
    else:
        print(f"Vector store {store_name} does not exist.")
        return []
