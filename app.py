import json
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load JSON dataset
with open("course_faqs.json", "r", encoding="utf-8") as file:
    data = json.load(file)

faqs = data["FAQs"]
questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert questions to embeddings
question_embeddings = model.encode(questions, convert_to_numpy=True)

# Build FAISS index for fast retrieval
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Initialize FastAPI
app = FastAPI()

# Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str

# Chatbot function
def chatbot_response(user_query):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    _, top_match = index.search(query_embedding, 1)  # Find closest question
    best_match_index = top_match[0][0]
    
    if best_match_index >= 0 and best_match_index < len(answers):
        return answers[best_match_index]
    return "I'm sorry, I don't understand your question."

# API Route
@app.post("/chat")
async def chat(request: ChatRequest):
    response = chatbot_response(request.message)
    return {"response": response}

#specific port
if name == "main":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)

