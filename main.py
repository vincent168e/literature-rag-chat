from dotenv import load_dotenv
from src.workflow import setup_rag_chain, continual_chat

load_dotenv()

def main():
    try:
        rag_chain = setup_rag_chain()
        continual_chat(rag_chain)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()