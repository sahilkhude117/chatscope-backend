import openai
from typing import List
from langchain.schema import Document
import os

class ChatService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI()
    
    async def generate_response(self, query: str, context_docs: List[Document]) -> str:
        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt
        prompt = f"""
        You are a helpful AI assistant. Answer the user's question based on the provided context.
        If the answer cannot be found in the context, say so politely.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"