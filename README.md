### stream.py  

   - Handles the Streamlit-based chatbot interface.  
   - Takes user queries, fetches relevant data from Weaviate, and generates answers using Llama3.  

### main.py  

   - Serves as the central controller for NCERTScraper, DocumentProcessor, and QueryProcessor.  

## How to Run This Project:  

   - **Set Up the Environment** – Install all required dependencies and ensure necessary Python packages are available. Configure environment variables correctly to access external services like Weaviate.  

   - **Scrape and Process Data** – The NCERTScraper extracts content from NCERT textbooks, while DocumentProcessor breaks it down into smaller chunks for better retrieval. These processed chunks are stored in Weaviate for efficient semantic search.  

   - **Launch the Chatbot UI** – The Streamlit-based interface lets users ask questions. QueryProcessor fetches relevant context from Weaviate, and Llama3 generates responses based on the retrieved data.  

   - **Evaluate Performance** – The evaluation script compares AI-generated responses with expected answers using BLEU, ROUGE, and BERTScore to measure accuracy.  
