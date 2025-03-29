import os
import shutil
import ollama
import PyPDF2
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import chatbot_response
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

def summarize_text(text):
    """Summarizes the extracted text into a concise paragraph."""
    prompt = (f"Summarize the following medical report concisely into a paragraph "
              f"(no text other than the summary itself, start the paragraph with the "
              f"date of the medical record), highlighting key diagnoses, symptoms, "
              f"treatments, and outcomes. Ensure clarity and medical accuracy while "
              f"keeping it within 6 to 8 lines.\n\n{text}")
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully", "filename": file.filename}

@app.get("/files")
async def list_files():
    files = os.listdir(UPLOAD_DIR)
    return {"files": files}

class FileDeleteRequest(BaseModel):
    filename: str

@app.delete("/delete")
async def delete_file(request: FileDeleteRequest):
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "File deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="File not found")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_chatbot(request: QueryRequest):
    response = chatbot_response(request.query)
    return {"response": response}

UPLOAD_DIR = "./uploads"
SUMMARY_FILE = "summaries.json"

@app.get("/check")
def check_filenames():
    try:
        with open(SUMMARY_FILE, "r") as f:
            summaries_data = json.load(f)

        if not isinstance(summaries_data, dict) or "summaries" not in summaries_data:
            return {"error": f"Unexpected format in {SUMMARY_FILE}, missing 'summaries' key."}

        summaries = summaries_data["summaries"]

        if not isinstance(summaries, list):
            return {"error": f"Unexpected format in {SUMMARY_FILE}, 'summaries' should be a list."}

        summary_filenames = {entry["filename"] for entry in summaries if isinstance(entry, dict) and "filename" in entry}

    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": f"Error reading {SUMMARY_FILE}: {str(e)}"}

    try:
        actual_filenames = {f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))}
    except FileNotFoundError:
        return {"error": f"Uploads directory '{UPLOAD_DIR}' not found"}

    missing_in_uploads = summary_filenames - actual_filenames
    missing_in_summary = actual_filenames - summary_filenames

    if missing_in_uploads:
        summaries_data["summaries"] = [entry for entry in summaries if entry["filename"] not in missing_in_uploads]
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summaries_data, f, indent=4)

    if missing_in_summary:
        for pdf_file in missing_in_summary:
            pdf_path = os.path.join(UPLOAD_DIR, pdf_file)
            extracted_text = extract_text_from_pdf(pdf_path)
            summary = summarize_text(extracted_text) if extracted_text else "No text extracted"
            summaries_data["summaries"].append({"filename": pdf_file, "summary": summary})

        with open(SUMMARY_FILE, "w") as json_file:
            json.dump(summaries_data, json_file, indent=4)


    return {
        "removed_from_summaries": list(missing_in_uploads),
        "missing_in_summary": list(missing_in_summary),
        "all_matched": len(missing_in_uploads) == 0 and len(missing_in_summary) == 0
    }

@app.get("/summarize")
async def summarize_uploaded_files():
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    summaries = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(UPLOAD_DIR, pdf_file)
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if extracted_text:
            summary = summarize_text(extracted_text)
            summaries.append({"filename": pdf_file, "summary": summary})
        else:
            summaries.append({"filename": pdf_file, "summary": "No text extracted"})
    
    with open("summaries.json", "w") as json_file:
        json.dump({"summaries": summaries}, json_file, indent=4)

    return {"success": "Saved to summaries.json successfully"}