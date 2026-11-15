# GenAI RAG Assistant: Technical Documentation Q&A

A Retrieval-Augmented Generation (RAG) pipeline built in Python to perform context-aware semantic search and question-answering on technical documentation (specifically, my M.Sc. Computer Science Thesis).

## Features
* **Semantic Search:** Converts user queries into vector embeddings using Cosine Similarity to find the most mathematically relevant document chunks.
* **Chunking:** Implements recursive character splitting (1000-token chunks with 200-token overlap) to maintain semantic context boundaries.
* **Source Citations:** Mitigates LLM hallucination by forcing the model to cite the exact page numbers from the source document used to generate the answer.

## Quick Start
Clone the project and install dependencies:
```bash
git clone https://github.com/kkgergov/genai-rag-assistant.git
cd genai-rag-assistant
pip install -r requirements.txt
```

Place your target PDF in the root directory and create a `.env` file in the root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

Then run the Assistant:
```bash
python main.py
```
