# %%
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from datasets import load_dataset
import numpy as np

# Setup logging
log_filename = "./finaltest/summarized_evaluation.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Assign a single device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
target_token = 3000
# Load the summarization model and tokenizer
summarizer_name = "google/long-t5-tglobal-large"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name).to(device)

# Load the ChatGLM2 model and tokenizer
model_name = "THUDM/chatglm2-6b-32k"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
model = model.half().to(device)  # Use FP16 for efficiency

logging.info("Summarization and ChatGLM2-6B-32k models loaded successfully on a single device.")


dataset = load_dataset("THUDM/LongBench", "multifieldqa_en", split="test")

logging.info("Dataset loaded successfully.")

avg_context_length = sum(len(item['context'].split()) for item in dataset) / len(dataset)
max_context_length = max(len(item['context'].split()) for item in dataset)
avg_input_length = sum(len((item['input'] + " " + item['context']).split()) for item in dataset) / len(dataset)

logging.info(f"Average Context Length: {avg_context_length:.2f} words")
logging.info(f"Average Input Length: {avg_input_length:.2f} words")
logging.info(f"Maximum Context Length: {max_context_length:.2f} words")

# %%
# Function to summarize context and calculate compression percentage
def summarize_context(context):
    original_length = len(context.split())
    #target_length = max(1, int(original_length * compression_rate))  
    inputs = summarizer_tokenizer(context, return_tensors="pt", max_length=16384, truncation=True).to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        summary_ids = summarizer_model.generate(
            inputs["input_ids"],
            max_length=target_token,
            min_length=1000,  # Ensure the summary isn't too short
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    compressed_length = len(summary.split())
    actual_compression_rate = compressed_length / original_length
    logging.info(
        f"Original Length: {original_length} words, Compressed Length: {compressed_length} words, Compression: {actual_compression_rate * 100:.2f}%"
    )
    return summary, actual_compression_rate * 100

# Function to get an answer from the model
def get_answer(question, context):
    summarized_context, compression_percentage = summarize_context(context)
    input_text = f"Question: {question}\nContext: {summarized_context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    
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
    return answer, latency, compression_percentage

# Evaluation metrics
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
compression_percentages = []

logging.info("Starting evaluation on the dataset.")

# Evaluate each example in the dataset and log compression percentages
for idx, item in enumerate(dataset):
    question = item["input"]
    context = item["context"]
    true_answer = item["answers"][0] if "answers" in item and item["answers"] else ""
    predicted_answer, latency, compression_percentage = get_answer(question, context)
    true_answers.append(true_answer)
    predicted_answers.append(predicted_answer)
    latencies.append(latency)
    compression_percentages.append(compression_percentage)
    logging.info(f"Item {idx + 1}/{len(dataset)} processed - Compression: {compression_percentage:.2f}%, Latency: {latency:.4f} seconds")

# Calculate evaluation metrics
f1 = calculate_f1_score(predicted_answers, true_answers)

# Compute latency statistics
avg_latency = np.mean(latencies)
max_latency = np.max(latencies)
min_latency = np.min(latencies)
avg_compression = np.mean(compression_percentages)

# Log results
logging.info("Evaluation completed.")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Average Latency: {avg_latency:.4f} seconds")
logging.info(f"Max Latency: {max_latency:.4f} seconds")
logging.info(f"Min Latency: {min_latency:.4f} seconds")
logging.info(f"Average Compression Percentage: {avg_compression:.2f}%")
logging.info(f"Target Token: {target_token}%")
print("Evaluation completed. Check the log file for detailed results.")
# %%
