"""Ask questions about thesis.pdf with a local, persistent RAG index.

Setup:
1. Install dependencies: ``pip install -r requirements.txt``
2. Create a ``.env`` file containing ``OPENAI_API_KEY=your-api-key``.
3. Place ``thesis.pdf`` next to this file and run: ``python main.py``
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


PDF_PATH = Path("thesis.pdf")
CHROMA_PATH = "./chroma_db"


def build_chain():
    """Load the thesis, index its chunks, and return a retrieval chain."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set in the environment or .env file.")

    documents = PyPDFLoader(str(PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the supplied context. "
                "If the answer is not in the context, say you do not know.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(ChatOpenAI(model="gpt-4o-mini"), prompt)
    return create_retrieval_chain(vector_store.as_retriever(), document_chain)


def main():
    load_dotenv()

    try:
        retrieval_chain = build_chain()
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return
    except ValueError as error:
        print(f"Error: {error}")
        return
    except Exception as error:
        print(f"Error initializing the RAG pipeline: {error}")
        return

    print("RAG assistant ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = retrieval_chain.invoke({"input": question})
            print(f"\nAnswer:\n{result['answer']}")

            sources = result.get("context", [])
            if sources:
                print("\nSources:")
                for document in sources:
                    page = document.metadata.get("page")
                    page_label = page + 1 if isinstance(page, int) else page
                    print(f"- Page {page_label}")
            else:
                print("\nSources: No source documents were retrieved.")
        except Exception as error:
            print(f"Error answering question: {error}")


if __name__ == "__main__":
    main()
