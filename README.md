# Document Question Answering System (RAG)

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries based on the contents of a PDF document. The application combines semantic search with Google's Gemini Large Language Model to retrieve relevant information from the document and generate accurate, context-aware responses.

The system processes a PDF document, converts its contents into vector embeddings, stores them in a FAISS vector database, and retrieves the most relevant information whenever a user asks a question.

---

## Features

- PDF document loading and processing
- Automatic text chunking
- Semantic embeddings using Google Gemini
- FAISS vector database for efficient similarity search
- Context-aware question answering
- Interactive command-line interface
- Environment variable support for secure API key management

---

## Technologies Used

- Python
- LangChain
- Google Gemini API
- FAISS
- PyPDF
- Python Dotenv

---

## Project Structure

```text
Week7_RAG/
│
├── app.py
├── sample.pdf
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
└── venv/
```

---


## How It Works

1. Loads the PDF document.
2. Extracts text from each page.
3. Splits the text into smaller chunks.
4. Generates vector embeddings using Google Gemini.
5. Stores embeddings in a FAISS vector database.
6. Accepts a user's question.
7. Retrieves the most relevant document chunks.
8. Sends the retrieved context to Gemini.
9. Generates and displays the final answer.

---

## Example Questions

- What is your CGPA?
- What are your technical skills?
- What projects have you completed?
- What is your email address?
- Which college are you studying in?
- What programming languages do you know?

---

## Dependencies

```text
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
google-genai
faiss-cpu
pypdf
python-dotenv
```

---

## Future Enhancements

- Support for multiple PDF documents
- Web-based user interface using Streamlit or Flask
- Conversation memory
- Source citation for retrieved answers
- Support for DOCX and TXT documents

---

## Author

**Shrirang Atul Badkas**

Master of Computer Applications (MCA)  
Dr. D. Y. Patil Institute of Management and Research, Pune

---


