import requests
from sentence_transformers import SentenceTransformer

class QueryProcessor:
    
    def __init__(self, weaviate_client, collection_name="DocumentChunks", embedding_model_name='all-MiniLM-L6-v2'):
        """Initialize with weaviate client and collection name."""
        self.client = weaviate_client
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"QueryProcessor initialized with collection: {collection_name}")
    
    def get_embedding(self, text):
        """Generate embeddings locally using sentence-transformers."""
        return self.embedding_model.encode(text).tolist()
    
    def query_ollama(self, prompt, model="llama3"):
        """Send a query to Ollama and return the response."""
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response generated")
        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return f"Error: {str(e)}"
    
    def process_query(self, query, limit=3):
        """Process a query and return generated answer."""
        try:
            # Generate embedding for query
            query_embedding = self.get_embedding(query)
            
            # Get collection
            documents_collection = self.client.collections.get(self.collection_name)
            
            # Perform vector search
            results = documents_collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit
            )
            
            # Process results
            context = "\n\n".join([obj.properties["text"] for obj in results.objects])
 
            # Generate answer using Ollama
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
            
            # Query Ollama
            ollama_response = self.query_ollama(ollama_prompt)

            return {
                "query": query,
                "context": context,
                "response": ollama_response
            }
        
        except Exception as e:
            print(f"Error during query: {e}")
            return {"error": str(e)}

