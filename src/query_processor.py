import requests
import os
from dotenv import load_dotenv
from document_processor import WeaviateManager
from document_processor import EmbeddingModel
 
load_dotenv()
 
class QueryProcessor:
    def __init__(self, weaviate_client=None):
        if weaviate_client:
            self.client = weaviate_client
            self.embedding_model = EmbeddingModel()
        else:
            self.weaviate_manager = WeaviateManager()
            self.client = self.weaviate_manager.client
            self.embedding_model = EmbeddingModel()
            
        self.collection_name = os.getenv("WEAVIATE_COLLECTION", "DocumentChunks")
    
 
    
    def query_ollama(self, prompt, model=os.getenv("OLLAMA_MODEL")):
        url = os.getenv("OLLAMA_URL")
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", f"Error: Unexpected API response format - {result}")
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
    
    def process_query(self, query, limit=int(os.getenv("LIMIT", 3))):
        try:
            query_embedding = self.embedding_model.get_embedding(query)
            
            documents_collection = self.client.collections.get(self.collection_name)
            response = documents_collection.query.near_vector(
                near_vector=query_embedding, limit=limit
            )
            
            if not response.objects:
                return {"query": query, "context": "No relevant documents found.", "response": "No data available."}
            
            context = "\n\n".join([obj.properties["text"] for obj in response.objects])
            ollama_prompt = f"""
            Answer this question: {query}
            
            Using this context from the document:
            {context}
            
            Provide a clear and concise answer based only on the information in the context.
            Use information striclty from the context provided only.

            If the context does not provide relevant answer, tell that context does not contain this information
            give short answer from your own memory but explicitly tell if its from your side.
            Only answer the question given in the prompt. Do not answer question that may be present in the contexts.
            Do not give much lengthy answer, make it more precise.
            """
            ollama_response = self.query_ollama(ollama_prompt)
            return {"query": query, "context": context, "response": ollama_response}
        except Exception as e:
            return {"error": str(e)}
 