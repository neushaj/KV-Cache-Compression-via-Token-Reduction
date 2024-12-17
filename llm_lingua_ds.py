# %%
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from llmlingua import PromptCompressor

# Initialize logging
log_filename = "./finaltest/llm_lingua_evaluation.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Clear GPU memory
torch.cuda.empty_cache()

# Load the model and tokenizer
model_name = "THUDM/chatglm2-6b-32k"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

logging.info("Model and tokenizer loaded successfully.")

# Load the dataset
dataset = load_dataset("THUDM/LongBench", "multifieldqa_en", split="test")

logging.info("Dataset Multifieldqa loaded successfully.")

avg_context_length = sum(len(item['context'].split()) for item in dataset) / len(dataset)
max_context_length = max(len(item['context'].split()) for item in dataset)
avg_input_length = sum(len((item['input'] + " " + item['context']).split()) for item in dataset) / len(dataset)

logging.info(f"Average Context Length: {avg_context_length:.2f} words")
logging.info(f"Average Input Length: {avg_input_length:.2f} words")
logging.info(f"Maximum Context Length: {max_context_length} words")

# Device setup
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.half().to(device)  # Use FP16 for efficiency
target_token = 10000
logging.info(f"Using device: {device}")

# Initialize LLMLingua
llm_lingua = PromptCompressor()

logging.info("LLMLingua PromptCompressor initialized.")
#%%
# Helper function with LLMLingua compression
def get_answer(question, context):
    compressed_context = llm_lingua.compress_prompt(
    context,
    instruction="",
    question="",
    target_token=target_token,
    condition_compare=True,
    condition_in_question="after",
    rank_method="llmlingua",
    use_sentence_level_filter=False,
    context_budget="+100",
    dynamic_context_compression_ratio=0.9,
    reorder_context="sort",
    )

    logging.info(
        f"Original Length: {compressed_context['origin_tokens']} words, Compressed Length: {compressed_context['compressed_tokens']} words, "
        f"Compression: {compressed_context['rate']}"
    )

    prompt = f"Question: {question}\nContext: {compressed_context['compressed_prompt']}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=16384)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure tensors are on the correct device
    
    start_time = time.time()  # Start timing
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    end_time = time.time()  # End timing
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()  # Post-process to extract the answer part
    latency = end_time - start_time
    return answer, latency, float(compressed_context['rate'].strip('%'))

# Function to calculate F1 score
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

# Evaluate on the full dataset with compression
true_answers = []
predicted_answers_compressed = []
latencies = []
compression_percentages = []

logging.info("Starting evaluation on the dataset.")

for idx, item in enumerate(dataset):
    question = item["input"]
    context = item["context"]
    true_answer = item["answers"][0] if "answers" in item and item["answers"] else ""
    predicted_answer, latency, compression_percentage = get_answer(question, context)
    true_answers.append(true_answer)
    predicted_answers_compressed.append(predicted_answer)
    latencies.append(latency)
    compression_percentages.append(compression_percentage)
    logging.info(
        f"Processed item {idx + 1}/{len(dataset)} - Compression: {compression_percentage:.2f}%, "
        f"Latency: {latency:.4f} seconds."
    )

# %%
# Compute evaluation metrics
f1_compressed = calculate_f1_score(predicted_answers_compressed, true_answers)

# Compute latency statistics
avg_latency = np.mean(latencies)
max_latency = np.max(latencies)
min_latency = np.min(latencies)
avg_compression = np.mean(compression_percentages)

# Log results
logging.info("Evaluation completed.")
logging.info(f"F1 Score: {f1_compressed:.4f}")
logging.info(f"Average Latency: {avg_latency:.4f} seconds")
logging.info(f"Max Latency: {max_latency:.4f} seconds")
logging.info(f"Min Latency: {min_latency:.4f} seconds")
logging.info(f"Average Compression Percentage: {avg_compression:.2f}%")
logging.info(f"Target Token: {target_token}")

print("Evaluation completed. Check the log file for detailed results.")
