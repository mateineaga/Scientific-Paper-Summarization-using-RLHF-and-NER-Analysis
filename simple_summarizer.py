import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, PegasusForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# from bert_score import score as bert_score
import matplotlib.pyplot as plt

print("Starting script...")

print("Downloading model...", datetime.now())
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(
    "cuda"
)
print("Downloading tokenizer...", datetime.now())
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

with open(
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/train_windows.json",
    "r",
) as f:
    data = json.load(f)


def summarize_window(window_text):
    inputs = tokenizer(
        window_text, truncation=True, padding=True, return_tensors="pt"
    ).to("cuda")
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=200,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded_summary = tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded_summary[0]


def evaluate_metrics(reference, summary):
    results = {}

    # ROUGE Scores
    rouge_scorer_instance = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rouge_scores = rouge_scorer_instance.score(reference, summary)
    results["rouge"] = {
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
    }

    # BLEU Score with Smoothing
    smoothing_function = SmoothingFunction().method1  # Apply smoothing function
    bleu_score = sentence_bleu(
        [reference.split()], summary.split(), smoothing_function=smoothing_function
    )
    results["bleu"] = bleu_score

    return results


evaluation_results = []

for article in data:
    article_id = article["id"]
    sections = article.get("windows", [])
    summary = article["summary"]

    summary_windows = []
    for i, window_text in enumerate(sections):
        window_summary = summarize_window(window_text)
        summary_windows.append(window_summary)

    final_summary = " ".join(summary_windows)

    print(f"Final summary for {article_id} done. {datetime.now()}")

    # summary_output_filename = f"articol-{article_id}-summary.txt"
    # with open(summary_output_filename, "w") as summary_file:
    #     summary_file.write(final_summary)

    reference_text = summary
    metrics = evaluate_metrics(reference_text, final_summary)
    evaluation_results.append({"article_id": article_id, "metrics": metrics})

    # print(
    #     f"Final summary saved for article {article_id} in {summary_output_filename}")

# Save metrics to JSON file
with open("evaluation_results.json", "w") as metrics_file:
    json.dump(evaluation_results, metrics_file, indent=4)

# Generate and display a graph
rouge1_scores = [result["metrics"]["rouge"]["rouge1"] for result in evaluation_results]
rouge2_scores = [result["metrics"]["rouge"]["rouge2"] for result in evaluation_results]
rougeL_scores = [result["metrics"]["rouge"]["rougeL"] for result in evaluation_results]
bleu_scores = [result["metrics"]["bleu"] for result in evaluation_results]
# bert_f1_scores = [result["metrics"]["bert"]["f1"]
#                   for result in evaluation_results]

plt.figure()
plt.plot(rouge1_scores, label="ROUGE-1")
plt.plot(rouge2_scores, label="ROUGE-2")
plt.plot(rougeL_scores, label="ROUGE-L")
plt.plot(bleu_scores, label="BLEU")
# plt.plot(bert_f1_scores, label="BERT F1")
plt.xlabel("Article Index")
plt.ylabel("Scores")
plt.title("Evaluation Metrics for Summaries")
plt.legend()
plt.savefig("evaluation_metrics.png")
plt.show()
