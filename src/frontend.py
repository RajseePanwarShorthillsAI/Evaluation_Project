import streamlit as st
import weaviate
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
 
def load_environment():
    """Load environment variables."""
    load_dotenv()
 
def initialize_weaviate_client():
    """Connect to the Weaviate cloud instance."""
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_RESTURL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_ADMIN"))
    )
 
def initialize_embedding_model():
    """Initialize the sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")
 
def query_weaviate(client, query_embedding, collection_name="DocumentChunks", limit=3):
    """Perform vector search in Weaviate."""
    documents_collection = client.collections.get(collection_name)
    results = documents_collection.query.near_vector(near_vector=query_embedding, limit=limit)
    return "\n\n".join([obj.properties["text"] for obj in results.objects])
 
def query_ollama(prompt, model="llama3"):
    """Send a query to Ollama for response generation."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "No response generated")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
 
def main():
    """Main function to run the Streamlit app."""
    load_environment()
    st.set_page_config(page_title="Weaviate Q&A", layout="wide")
    st.title("📚 AI-Powered Q&A for NCERT History (Class 6-10)")
    # Initialize session state for history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "answer_history" not in st.session_state:
        st.session_state.answer_history = [] 
    client = initialize_weaviate_client()
    embedding_model = initialize_embedding_model()
    st.sidebar.title("📝 Previous Queries")
    for i, query in enumerate(st.session_state.query_history):
        with st.sidebar.expander(f"{query}"):
            st.write(st.session_state.answer_history[i])

    query = st.text_input("🔍 Ask a question:")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching for the best answer..."):
                query_embedding = embedding_model.encode(query).tolist()
                context = query_weaviate(client, query_embedding)
                ollama_prompt = f"""
                Answer this question: {query}
                Using this context from the document:
                {context}
                Provide a clear and concise answer based only on the information in the context.
                """
                ollama_response = query_ollama(ollama_prompt)
            # Store query and answer history
            st.session_state.query_history.append(query)
            st.session_state.answer_history.append(ollama_response)
            st.subheader("🤖 AI Answer:")
            st.markdown(f"""<div style="background-color:#f9f9f9;padding:10px;border-radius:5px;">
                        {ollama_response}</div>""", unsafe_allow_html=True)
            with st.expander("📄 Show Retrieved Context"):
                st.write(context)
        else:
            st.warning("⚠️ Please enter a question!")
    client.close()
 
if __name__ == "__main__":
    main()

 