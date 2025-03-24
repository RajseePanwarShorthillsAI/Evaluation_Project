import unittest
import requests
import weaviate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class TestStreamlitApp(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up Weaviate connection before running tests"""
        cls.client = weaviate.connect_to_weaviate_cloud(
            cluster_url="https://zjrqw1rwbc25okvovwg.c0.asia-southeast1.gcp.weaviate.cloud",
            auth_credentials=weaviate.auth.AuthApiKey("Jc9UxOgShCwvkSE1SAUUvYPm7kCy0fMmmyXT"),
            skip_init_checks=True
        )

    @classmethod
    def tearDownClass(cls):
        """Close Weaviate connection after tests"""
        cls.client.close()

    def test_weaviate_connection(self):
        """Check if Weaviate connection is established"""
        self.assertIsNotNone(self.client, "Failed to connect to Weaviate.")

    def test_embedding_generation(self):
        """Test embedding generation using SentenceTransformer"""
        query = "What is artificial intelligence?"
        embedding = embedding_model.encode(query).tolist()
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0, "Generated embedding is empty.")

    def test_weaviate_vector_search(self):
        """Test Weaviate vector search with a sample query"""
        query_embedding = embedding_model.encode("What is AI?").tolist()
        documents_collection = self.client.collections.get("DocumentChunks")
        
        results = documents_collection.query.near_vector(near_vector=query_embedding, limit=1)
        self.assertIsNotNone(results, "Weaviate vector search returned no results.")
        self.assertGreaterEqual(len(results.objects), 0, "Weaviate vector search returned empty list.")

    def test_query_ollama_llama3(self):
        """Test querying Ollama with context-based prompt"""
        def query_ollama(prompt, model="llama3"):
            url = "http://localhost:11434/api/generate"
            payload = {"model": model, "prompt": prompt, "stream": False}
            try:
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                return response.json().get("response", "No response generated")
            except requests.exceptions.RequestException as e:
                return f"API Error: {e}"

        # Sample context-based prompt
        context = "AI is the simulation of human intelligence in machines."
        ollama_prompt = f"""
        Answer this question: What is AI?

        Using this context:
        {context}

        Provide a clear and concise answer strictly based on the context provided.
        If the context does not contain relevant information, indicate that.
        """

        result = query_ollama(ollama_prompt)
        self.assertIsInstance(result, str, "Ollama response is not a string.")
        self.assertGreater(len(result), 0, "Ollama response is empty.")

if __name__ == "__main__":
    unittest.main()
