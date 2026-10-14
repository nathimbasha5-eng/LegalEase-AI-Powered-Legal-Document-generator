import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="LegalEase API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentRequest(BaseModel):
    doc_type: str
    parties: str
    terms: str
    jurisdiction: str

@app.post("/api/generate-document")
async def generate_document(req: DocumentRequest):
    try:
        prompt = f"""
        Draft a formal legal document based on the following details:
        - Document Type: {req.doc_type}
        - Parties Involved: {req.parties}
        - Key Terms & Conditions: {req.terms}
        - Jurisdiction: {req.jurisdiction}
        
        Ensure proper legal formatting, standard clauses, and a professional, binding tone.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert legal assistant specialized in drafting clear, robust legal agreements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        document_text = response.choices[0].message.content
        return {"success": True, "document": document_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)