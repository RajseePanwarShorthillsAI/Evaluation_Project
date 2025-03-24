import pandas as pd
import google.generativeai as genai
from utils import get_api_key
import time
import json
import re
import os

class CSVLoader:
    """Handles loading and preprocessing of the CSV file."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_csv()
    
    def load_csv(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        df = pd.read_csv(self.file_path)
        df.columns = df.columns.str.strip()  # Clean column names, strip spaces
        df = df.dropna(subset=["Generated Answer", "Ground Truth"])  # Drop empty rows
        return df

class GeminiEvaluator:
    """Handles API calls to Gemini for evaluation."""
    def __init__(self):
        genai.configure(api_key=get_api_key())
        self.model = genai.GenerativeModel(os.getenv('GENAI_MODEL'))
    
    @staticmethod
    def clean_json_response(response_text):
        response_text = re.sub(r"```json\s*|\s*```", "", response_text.strip())
        try:
            scores = json.loads(response_text)
            return {
                "Faithfulness": float(scores.get("Faithfulness", 0)),
                "Precision": float(scores.get("Precision", 0)),
                "Recall": float(scores.get("Recall", 0)),
                "Relevancy": float(scores.get("Relevancy", 0)),
            }
        except json.JSONDecodeError:
            print(f"JSON Decode Error: {response_text}")
        return {"Faithfulness": 0, "Precision": 0, "Recall": 0, "Relevancy": 0}
    
    def evaluate(self, question, generated_answer, ground_truth, contexts):
        prompt = f"""
        You are evaluating a question-answering system based on four key metrics...
        Provide scores (0-1) in **valid JSON format** as follows:
        ```json
        {{"Faithfulness": 0.85, "Precision": 0.5, "Recall": 0.7, "Relevancy": 0.3}}
        ```
        Question: {question}
        Generated Answer: {generated_answer}
        Ground Truth: {ground_truth}
        Contexts: {contexts}
        """
        try:
            response = self.model.generate_content(prompt)
            return self.clean_json_response(response.text)
        except Exception as e:
            print(f"Error processing Gemini API: {e}")
            return {"Faithfulness": 0, "Precision": 0, "Recall": 0, "Relevancy": 0}

class ResultManager:
    """Manages result storage and incremental saving."""
    def __init__(self, result_file):
        self.result_file = result_file
        self.processed_questions = set()
        self.existing_results = self.load_existing_results()
    
    def load_existing_results(self):
        if os.path.exists(self.result_file) and os.path.getsize(self.result_file) > 0:
            try:
                df = pd.read_csv(self.result_file)
                if "Generated Answer" in df.columns:
                    self.processed_questions = set(df["Generated Answer"].dropna().str.strip())
                return df
            except Exception as e:
                print(f"Error reading results file: {e}")
        return pd.DataFrame()
    
    def save_results(self, new_results):
        new_df = pd.DataFrame(new_results)
        combined_df = pd.concat([self.existing_results, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["Generated Answer"])
        combined_df.to_csv(self.result_file, index=False)
        print(f"Saved {len(new_results)} new results.")

# === Main Execution ===
file_path = "ragas_testing.csv"
result_file = "testans_results.csv"

data_loader = CSVLoader(file_path)
evaluator = GeminiEvaluator()
result_manager = ResultManager(result_file)

results = []

for index, row in data_loader.df.iterrows():
    generated_answer = str(row["Generated Answer"]).strip()
    ground_truth = str(row["Ground Truth"]).strip()
    question = str(row.get("Question", "")).strip()
    
    if not generated_answer or generated_answer in result_manager.processed_questions:
        print(f"Skipping already processed or empty answer at index {index}")
        continue
    
    references = [str(row[f"Context {i}"].strip()) for i in range(1, 5) if f"Context {i}" in row and pd.notna(row[f"Context {i}"])]
    print(f"Processing question {index + 1}...")
    
    scores = evaluator.evaluate(question, generated_answer, ground_truth, references)
    
    results.append({
        "Question": question,
        "Generated Answer": generated_answer,
        "Ground Truth": ground_truth,
        "Faithfulness": round(scores["Faithfulness"], 2),
        "Precision": round(scores["Precision"], 2),
        "Recall": round(scores["Recall"], 2),
        "Relevancy": round(scores["Relevancy"], 2)
    })
    
    if len(results) % 5 == 0:
        result_manager.save_results(results)
        results.clear()  # Reset results list after saving
    
    time.sleep(5)

if results:
    result_manager.save_results(results)
else:
    print("⚠️ No new rows were processed.")
