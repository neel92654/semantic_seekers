# semantic_seekers
We are the team Semantic Seekers prepared an AI Assistant built with LangChain and local LLMs. It answers questions strictly from uploaded PDFs using retrieval-augmented generation, validates responses to prevent hallucinations, and provides accurate, source-grounded insights with conversation history and analytics.

Here is a **clean, plain README.md** without icons or emojis, ready to **copy and paste**:

---

# Documentation-Aware AI Assistant

## Overview

This project is an AI-powered documentation assistant built using LangChain, vector databases, and local Large Language Models (LLMs). It answers user questions strictly based on uploaded documents (PDF/TXT), prevents hallucinations, validates responses, and provides source-grounded answers.


## Key Features

* Document-based question answering (PDF and TXT)
* Retrieval-Augmented Generation (RAG)
* Dual-LLM architecture with answer generation and validation
* Prevention of out-of-context and hallucinated answers
* Source-aware responses
* Support for multiple documents
* Persistent analytics with feedback tracking


## Tech Stack

* Python
* LangChain
* FAISS (Vector Database)
* HuggingFace Embeddings
* Ollama (Mistral LLM)
* Matplotlib

---

## How It Works

1. Documents are loaded and split into semantic chunks
2. Chunks are embedded and stored in a vector database
3. User queries retrieve relevant document context
4. The generator LLM produces an answer
5. The validator LLM verifies grounding against the document
6. Only validated answers are returned with source reference

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Documents

Place your PDF or TXT files inside the `data/` directory.

### Run the Assistant

```bash
python main.py
```


## Example Usage

```
You: What teams did Sachin Tendulkar play for?
Answer:
Sachin Tendulkar represented India internationally and Mumbai in domestic cricket.
Source: Sachin.pdf
```

If the question is not present in the documents:

```
Answer Not Available
This information is not present in the uploaded documents.
```

## Analytics

* Total number of questions asked
* Feedback distribution (good/bad/unknown)
* Question trends across sessions

## Use Cases

* Academic notes and textbooks
* Technical documentation
* Manuals and reports
* Knowledge-based systems

## Future Enhancements

* Page-level citations
* Confidence scoring
* Web interface or API
* Multi-user support

If you want this shortened further or tailored for a hackathon submission, let me know.

