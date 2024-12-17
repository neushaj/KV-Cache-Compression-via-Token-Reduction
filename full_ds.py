import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np

# Setup logging
log_filename = "full_ds_evaluation2.log"
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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.half().to(device)  # Use FP16 for efficiency

logging.info("Model and tokenizer for ChatGLM2-6B-32k loaded successfully.")

# Load the Qasper dataset
dataset = load_dataset("THUDM/LongBench", "multifieldqa_en", split="test")

logging.info("Dataset loaded successfully.")

# Calculate average context and input length
avg_context_length = sum(len(item['context'].split()) for item in dataset) / len(dataset)
max_context_length = max(len(item['context'].split()) for item in dataset)
avg_input_length = sum(len((item['input'] + " " + item['context']).split()) for item in dataset) / len(dataset)

logging.info(f"Average Context Length: {avg_context_length:.2f} words")
logging.info(f"Average Input Length: {avg_input_length:.2f} words")
logging.info(f"Maximum Context Length: {max_context_length} words")

# Function to get an answer from the model
def get_answer(question, context):
    # Combine question and context for ChatGLM2 input
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=13400)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Adjust token generation limit as needed
            temperature=0.7,
            top_k=50
        )
    end_time = time.time()
    
    # Decode the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    latency = end_time - start_time
    return answer, latency

# Evaluation metrics
def calculate_exact_match(predicted_answers, true_answers):
    matches = [1 if pred.strip().lower() == true.strip().lower() else 0 for pred, true in zip(predicted_answers, true_answers)]
    return 100.0 * sum(matches) / len(matches)

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

# Prepare lists for true and predicted answers, and latency tracking
true_answers = []
predicted_answers = []
latencies = []

logging.info("Starting evaluation on the Qasper dataset.")

# Evaluate each example in the dataset
for idx, item in enumerate(dataset):
    question = item["input"]
    context = item["context"]
    true_answer = item["answers"][0] if "answers" in item and item["answers"] else ""
    predicted_answer, latency = get_answer(question, context)
    true_answers.append(true_answer)
    predicted_answers.append(predicted_answer)
    latencies.append(latency)
    logging.info(f"Item {idx + 1}/{len(dataset)} processed - Latency: {latency:.4f} seconds")

# Calculate evaluation metrics
# exact_match = calculate_exact_match(predicted_answers, true_answers)
f1 = calculate_f1_score(predicted_answers, true_answers)

# Compute latency statistics
avg_latency = np.mean(latencies)
max_latency = np.max(latencies)
min_latency = np.min(latencies)

# Log results
logging.info("Evaluation completed.")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Average Latency: {avg_latency:.4f} seconds")
logging.info(f"Max Latency: {max_latency:.4f} seconds")
logging.info(f"Min Latency: {min_latency:.4f} seconds")

print("Evaluation completed. Check the log file for detailed results.")
