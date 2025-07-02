from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.chat_service import ChatService

load_dotenv()

app = FastAPI(title="ChatScope AI Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStore()
chat_service = ChatService()

class ChatRequest(BaseModel):
    query: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: list = []

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = f"uploaded_docs/{file.filename}"
        os.makedirs("uploaded_docs", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process PDF and add to vector store
        chunks = pdf_processor.process_pdf(file_path)
        vector_store.add_documents(chunks)
        
        return {"message": f"File {file.filename} uploaded and processed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get relevant documents
        relevant_docs = vector_store.search(request.query)
        
        # Generate response using RAG
        answer = await chat_service.generate_response(
            query=request.query,
            context_docs=relevant_docs
        )
        
        return ChatResponse(
            answer=answer,
            sources=[doc.metadata.get("source", "") for doc in relevant_docs]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "ChatScope AI Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)