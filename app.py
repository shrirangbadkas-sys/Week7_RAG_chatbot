import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS


# Load Environment Variables

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

print(" API Loaded Successfully")


# Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print("PDF Loaded")
print(f"Total Pages : {len(documents)}")


# Split PDF into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks : {len(chunks)}")


# Create Embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

print(" Embedding Model Loaded")


# Create FAISS Vector Store

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

print(" FAISS Vector Database Created")


# Load Gemini Chat Model

llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

print("Gemini Chat Model Loaded")


# Chatbot

print("\n" + "=" * 45)
print("      DOCUMENT QUESTION ANSWERING SYSTEM")
print("=" * 45)

while True:

    question = input("\nAsk a Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("\n Thank you for using the RAG Chatbot!")
        break

    # Retrieve similar chunks
    docs = vector_db.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an intelligent AI assistant.

Answer ONLY from the context below.

If the answer is available, answer in a complete sentence.

If the answer is not available in the context, reply:

"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    try:

        response = llm.invoke(prompt)

        print("\n" + "=" * 45)
        print("ANSWER")
        print("=" * 45)

        # Different versions of LangChain/SDK return responses differently
        if hasattr(response, "text") and response.text:
            print(response.text)

        elif hasattr(response, "content"):

            if isinstance(response.content, str):
                print(response.content)

            elif isinstance(response.content, list):

                printed = False

                for item in response.content:

                    if isinstance(item, dict):

                        if "text" in item:
                            print(item["text"])
                            printed = True
                            break

                    elif hasattr(item, "text"):
                        print(item.text)
                        printed = True
                        break

                if not printed:
                    print(response.content)

            else:
                print(response.content)

        else:
            print(response)

        print("=" * 45)

    except Exception as e:
        print("\n Error:")
        print(e)
