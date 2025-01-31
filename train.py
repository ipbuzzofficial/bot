import json
import faiss
import numpy as np
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

# Function to get the best matching answer
def chatbot_response(user_query):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    _, top_match = index.search(query_embedding, 1)
    best_match_index = top_match[0][0]
    return answers[best_match_index]

# Example Chatbot Interaction
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
