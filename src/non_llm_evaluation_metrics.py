import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

class Scorer:
    @staticmethod
    def compute_bleu(references, candidate):
        reference_tokens = [ref.split() for ref in references]
        candidate_tokens = candidate.split()
        chencherry = SmoothingFunction()
        return sentence_bleu(reference_tokens, candidate_tokens,
                             weights=(0.5, 0.5, 0, 0),
                             smoothing_function=chencherry.method1) * 100
    
    @staticmethod
    def compute_rouge(references, candidate):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []
        
        for ref in references:
            scores = scorer.score(ref, candidate)
            rouge_1_scores.append(scores['rouge1'].fmeasure * 100)
            rouge_2_scores.append(scores['rouge2'].fmeasure * 100)
            rouge_l_scores.append(scores['rougeL'].fmeasure * 100)
        
        return {
            "ROUGE-1": max(rouge_1_scores),
            "ROUGE-2": max(rouge_2_scores),
            "ROUGE-L": max(rouge_l_scores)
        }
    
    @staticmethod
    def compute_bertscore(references, candidate):
        scores = []
        for ref in references:
            _, _, F1 = score([candidate], [ref], lang="en", verbose=False)
            scores.append(F1.item() * 100)
        return max(scores)

class DataProcessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    
    def get_samples(self):
        for index, row in self.df.iterrows():
            generated_answer = row["Generated Answer"]
            ground_truth = row["Ground Truth"]
            references = [ground_truth] + [row["Context1"], row["Context2"], row["Context3"]]
            references = [ref for ref in references if pd.notna(ref)]
            yield index, generated_answer, references, row

class Evaluator:
    def __init__(self, file_path):
        self.processor = DataProcessor(file_path)
        self.evaluation_results = []
        self.total_scores = {"BLEU": 0, "ROUGE-1": 0, "ROUGE-2": 0, "ROUGE-L": 0, "BERTScore": 0}
        self.num_samples = len(self.processor.df)
    
    def evaluate(self):
        for index, generated_answer, references, row in self.processor.get_samples():
            print(f"Processing: {index}")
            
            bleu = Scorer.compute_bleu(references, generated_answer)
            rouge_scores = Scorer.compute_rouge(references, generated_answer)
            bertscore = Scorer.compute_bertscore(references, generated_answer)
            
            self.evaluation_results.append({
                "Generated Answer": generated_answer,
                "Ground Truth": row["Ground Truth"],
                "Context1": row["Context1"],
                "Context2": row["Context2"],
                "Context3": row["Context3"],
                "BLEU Score": bleu,
                "ROUGE-1": rouge_scores["ROUGE-1"],
                "ROUGE-2": rouge_scores["ROUGE-2"],
                "ROUGE-L": rouge_scores["ROUGE-L"],
                "BERTScore": bertscore
            })
            
            self.total_scores["BLEU"] += bleu
            self.total_scores["ROUGE-1"] += rouge_scores["ROUGE-1"]
            self.total_scores["ROUGE-2"] += rouge_scores["ROUGE-2"]
            self.total_scores["ROUGE-L"] += rouge_scores["ROUGE-L"]
            self.total_scores["BERTScore"] += bertscore
    
    def get_averages(self):
        return {key: self.total_scores[key] / self.num_samples for key in self.total_scores}
    
    def save_results(self, output_file="input_ques_ans.csv"):
        pd.DataFrame(self.evaluation_results).to_csv(output_file, index=False)
        print(f"\nEvaluation results saved to {output_file}")
    
    def run_evaluation(self):
        self.evaluate()
        averages = self.get_averages()
        self.save_results()
        
        print("\nOverall Evaluation Results:")
        for key, value in averages.items():
            print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    evaluator = Evaluator("que_answers.csv")
    evaluator.run_evaluation()
