import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient  

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("⚠️ Hugging Face API Key not found. Set 'HUGGINGFACEHUB_API_TOKEN' as an environment variable.")

def load_summaries():
    try:
        with open("summaries.json", "r") as f:
            data = json.load(f)
        return data.get("summaries", [])
    except FileNotFoundError:
        print("❌ summaries.json not found! FAISS will work without medical report data.")
        return []
    except json.JSONDecodeError:
        print("❌ Error reading summaries.json! Please check file integrity.")
        return []

summaries = load_summaries()

from langchain_core.documents import Document

from langchain_core.documents import Document  

def create_faiss_database(summaries):
    if not summaries:
        print("⚠️ No summaries available for FAISS retrieval.")
        return None

    docs = [
        Document(
            page_content=f"Filename: {summary['filename']}\nSummary: {summary['summary']}",
            metadata={"filename": summary["filename"]}  
        ) 
        for summary in summaries
    ]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_documents(split_docs, embeddings)



vector_db = create_faiss_database(summaries)
print("✅ FAISS Vector Database Created!")

huggingface_client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=HUGGINGFACE_API_KEY
)

print("✅ RAG Model Initialized!")


def chatbot_response(user_query):
    """Returns a response using FAISS first, then Hugging Face API if needed."""
    try:
        print("\n🔍 Querying FAISS...")
        retrieved_docs = vector_db.similarity_search(user_query, k=3) if vector_db else []
        context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""

        print("\n🔍 FAISS Retrieved Context:\n", context if context else "❌ No relevant FAISS data")

        if context:
            prompt_template = PromptTemplate.from_template(
                "You are a medical AI assistant that analyzes a user’s health reports over time to identify trends and provide meaningful insights. "
                "Explain medical terms in simple, easy-to-understand language."
                "You are talking to the user directly whose data u are analyzing"
                "You explain hard-to-understand medical terms in simple words (for a 10th-grade graduate) when needed."
                "If a term might be difficult, ask the user if they want an explanation before continuing."
                "If the available reports are insufficient to answer a query, rely on general medical knowledge to provide helpful guidance"
                "You are not a doctor and cannot provide medical diagnoses, prescribe medication, or replace professional medical advice. Always encourage users to consult their healthcare provider for medical decisions."
                "Ensure that all responses are clear, informative, and focused on the user’s health."
                "Your goal is to help users understand their health better and make informed decisions based on their medical history."
                "Use the provided medical summaries to detect patterns (e.g., a rising blood pressure trend) and alert the user to potential health risks..\n\n"
                "Context:\n{context}\n\nQuery: {query}"
            )

            formatted_prompt = prompt_template.format(context=context, query=user_query)

            rag_response = ask_huggingface(formatted_prompt)

            print(f"\n📝 FAISS Response: {rag_response}")

            if not rag_response.strip() or "not mentioned in the context" in rag_response.lower():
                print("\n⚠️ FAISS failed! Asking Hugging Face API...")
                return ask_huggingface(user_query)

            return rag_response  

        print("\n⚠️ FAISS had no useful answer. Asking Hugging Face API...")
        return ask_huggingface(user_query)

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def ask_huggingface(query):
    try:
        print("\n🌍 Querying Hugging Face API for:", query)

        response = huggingface_client.chat_completion(messages=[{"role": "user", "content": query}])

        if response and "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]
            print("\n🔹 Hugging Face Answer:", answer)
            return answer

        return "❌ Hugging Face API failed to generate an answer."

    except Exception as e:
        return f"⚠️ Hugging Face API Error: {str(e)}"
