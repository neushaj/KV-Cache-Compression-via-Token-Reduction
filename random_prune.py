import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from llmlingua import PromptCompressor

def calculate_f1_score(predicted_answers, true_answers):
    def precision_recall_f1(pred_tokens, true_tokens):
        common_tokens = set(pred_tokens) & set(true_tokens)
        if not common_tokens:
            return 0, 0, 0
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    f1_scores = []
    for pred, true in zip(predicted_answers, true_answers):
        pred_tokens = pred.split()
        true_tokens = true.split()
        _, _, f1 = precision_recall_f1(pred_tokens, true_tokens)
        f1_scores.append(f1)
    
    return 100.0 * sum(f1_scores) / len(f1_scores)

# Setup logging
log_filename = "/home/neusha/KVCache/Prj_NLP/finaltest/prunned_evaluation.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Load the model and tokenizer
model_name = "THUDM/chatglm2-6b-32k"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

target_tokens = 10000
model = model.half().to(device)  # Use FP16 for efficiency

logging.info("Model and tokenizer for ChatGLM2-6B-32k loaded successfully.")

# Load the Qasper dataset
dataset = load_dataset("THUDM/LongBench", "multifieldqa_en", split="test")

logging.info("Dataset loaded successfully.")

# Initialize LLMLingua PromptCompressor to use its tokenizer
llm_lingua = PromptCompressor()

# Function to prune context
def prune_context(context, target_tokens):
    tokens = llm_lingua.tokenizer(context)
    original_length = len(tokens['input_ids'])
    
    if original_length <= target_tokens:
        return context, original_length, original_length  # No pruning required
    
    selected_indices = sorted(np.random.choice(original_length, target_tokens, replace=False))
    pruned_tokens = [tokens['input_ids'][i] for i in selected_indices]
    pruned_context = llm_lingua.tokenizer.decode(pruned_tokens, skip_special_tokens=True)
    return pruned_context, original_length, len(pruned_tokens)

# Function to get an answer
def get_answer(question, context):
    pruned_context, original_length, compressed_length = prune_context(context, target_tokens)
    input_text = f"Question: {question}\nContext: {pruned_context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=13400)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_k=50
        )
    end_time = time.time()
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    latency = end_time - start_time
    compression_rate = 100 * (compressed_length / original_length)
    
    return answer, latency, original_length, compressed_length, compression_rate

# Evaluate each example in the dataset
true_answers = []
predicted_answers = []
latencies = []
compression_rates = []

logging.info("Starting evaluation on the Qasper dataset.")

for idx, item in enumerate(dataset):
    question = item["input"]
    context = item["context"]
    true_answer = item["answers"][0] if "answers" in item and item["answers"] else ""
    predicted_answer, latency, original_length, compressed_length, compression_rate = get_answer(question, context)
    true_answers.append(true_answer)
    predicted_answers.append(predicted_answer)
    latencies.append(latency)
    compression_rates.append(compression_rate)
    
    logging.info(
        f"Original Length: {original_length} tokens, Compressed Length: {compressed_length} tokens, "
        f"Compression: {compression_rate:.2f}%"
    )
    logging.info(
        f"Processed item {idx + 1}/{len(dataset)} - Compression: {compression_rate:.2f}%, "
        f"Latency: {latency:.4f} seconds."
    )

# Calculate evaluation metrics
f1 = calculate_f1_score(predicted_answers, true_answers)
avg_compression = np.mean(compression_rates)
avg_latency = np.mean(latencies)
max_latency = np.max(latencies)
min_latency = np.min(latencies)

# Log summary
logging.info("Evaluation completed.")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Average Compression Percentage: {avg_compression:.2f}%")
logging.info(f"Average Latency: {avg_latency:.4f} seconds")
logging.info(f"Max Latency: {max_latency:.4f} seconds")
logging.info(f"Min Latency: {min_latency:.4f} seconds")
logging.info(f"Target Token: {target_tokens}")

print("Evaluation completed. Check the log file for detailed results.")
