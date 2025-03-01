#!/usr/bin/env python3
import json
import time
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PegasusConfig,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    AdamW,
    pipeline
)
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
import matplotlib.pyplot as plt

nltk.download("wordnet")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# SECȚIUNEA DE HIPERPARAMETRI
# ==========================

# Antrenare
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.001

# Generare
MAX_LENGTH = 256
NUM_BEAMS = 6
EARLY_STOPPING = True

# ==========================
# CONFIGURARE MODEL ȘI TOKENIZER
# ==========================
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(
    "cuda"
)
# ==========================
# ÎNCĂRCARE ȘI CONFIGURARE DATE
# ==========================
train_path = (
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/train_windows.json"
)
val_path = (
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/val_windows.json"
)
test_path = (
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/test_windows.json"
)


def load_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


train_data = load_data(train_path)
train_data = train_data[:4227]
val_data = load_data(val_path)
val_data = val_data[:5]
test_data = load_data(test_path)
test_data = test_data[:5]


# ==========================
# FUNCȚII AUXILIARE
# ==========================
def evaluate_ner_detailed(reference, summary):
    """
    Perform a detailed NER evaluation comparing reference and summary entities.
    """
    reference_entities = ner_model(reference)
    summary_entities = ner_model(summary)

    # Extract entities with spans
    reference_spans = {
        (ent["start"], ent["end"], ent["entity"]) for ent in reference_entities
    }
    summary_spans = {
        (ent["start"], ent["end"], ent["entity"]) for ent in summary_entities
    }

    # Categorize matches
    perfect_matches = reference_spans.intersection(summary_spans)
    partial_matches = {
        (ref_start, ref_end, ref_label)
        for (ref_start, ref_end, ref_label) in reference_spans
        for (sum_start, sum_end, sum_label) in summary_spans
        if (
            ref_label == sum_label
            and (
                (ref_start <= sum_start < ref_end) or (
                    sum_start <= ref_start < sum_end)
            )
        )
    } - perfect_matches

    type_errors = {
        (ref_start, ref_end, ref_label)
        for (ref_start, ref_end, ref_label) in reference_spans
        for (sum_start, sum_end, sum_label) in summary_spans
        if ((ref_start, ref_end) == (sum_start, sum_end) and ref_label != sum_label)
    }

    spurious = summary_spans - reference_spans
    missing = reference_spans - summary_spans

    # Return results
    return {
        "perfect_matches": len(perfect_matches),
        "partial_matches": len(partial_matches),
        "type_errors": len(type_errors),
        "spurious": len(spurious),
        "missing": len(missing),
        "total_reference_entities": len(reference_spans),
        "total_summary_entities": len(summary_spans),
    }


ner_model = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
)


def evaluate_ner(reference, summary):
    """
    Perform a simple NER evaluation comparing reference and summary entities.
    """
    reference_entities = ner_model(reference)
    summary_entities = ner_model(summary)

    # Extract entity types for comparison
    reference_entity_types = {ent["entity"] for ent in reference_entities}
    summary_entity_types = {ent["entity"] for ent in summary_entities}

    # Overlap in detected entity types
    common_entities = reference_entity_types.intersection(summary_entity_types)

    return {
        "reference_entities": len(reference_entities),
        "summary_entities": len(summary_entities),
        "common_entities": len(common_entities),
        "entity_overlap": (
            len(common_entities) / len(reference_entity_types)
            if reference_entity_types
            else 0
        ),
    }


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
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [reference.split()], summary.split(), smoothing_function=smoothing_function
    )
    results["bleu"] = bleu_score

    # NER Analysis (Simple)
    ner_simple_results = evaluate_ner(reference, summary)
    results["ner_simple"] = ner_simple_results

    # NER Analysis (Detailed)
    ner_detailed_results = evaluate_ner_detailed(reference, summary)
    results["ner_detailed"] = ner_detailed_results

    return results


def summarize_window(window_text):
    inputs = tokenizer(
        window_text, truncation=True, padding=True, return_tensors="pt"
    ).to("cuda")
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_LENGTH,
        num_beams=NUM_BEAMS,
        early_stopping=EARLY_STOPPING,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded_summary = tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded_summary[0]


def generate_summary_from_windows(windows):
    summaries = []
    for idx, window_text in enumerate(windows):
        summary = summarize_window(window_text)
        summaries.append(summary)

    final_summary = " ".join(summaries)

    return final_summary


def evaluate_model(model, dataloader, device):
    model.eval()
    print('Starting evaluate mode!')
    evaluation_results = []  # Listă pentru stocarea rezultatelor per articol

    with torch.no_grad():
        for article in dataloader:
            article_id = article["id"]
            sections = article.get("windows", [])
            reference_summary = article["summary"]

            # Generăm rezumatul final pe baza ferestrelor
            summary_windows = []
            for i, window_text in enumerate(sections):
                window_summary = summarize_window(window_text)
                summary_windows.append(window_summary)

            final_summary = " ".join(summary_windows)

            # Calculăm metricile
            metrics = evaluate_metrics(reference_summary, final_summary)

            # Adăugăm rezultatele într-un dicționar per articol
            evaluation_results.append({
                "article_id": article_id,
                "metrics": metrics
            })

    # Calculăm metricile globale (agregate)
    rouge1_scores = [result["metrics"]["rouge"]["rouge1"]
                     for result in evaluation_results]
    rouge2_scores = [result["metrics"]["rouge"]["rouge2"]
                     for result in evaluation_results]
    rougeL_scores = [result["metrics"]["rouge"]["rougeL"]
                     for result in evaluation_results]
    bleu_scores = [result["metrics"]["bleu"] for result in evaluation_results]
    ner_entity_overlap = [result["metrics"]["ner_simple"]
                          ["entity_overlap"] for result in evaluation_results]

    # Calculăm media metricilor
    global_metrics = {
        "avg_rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "avg_rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "avg_rougeL": sum(rougeL_scores) / len(rougeL_scores),
        "avg_bleu": sum(bleu_scores) / len(bleu_scores),
        "avg_ner_entity_overlap": sum(ner_entity_overlap) / len(ner_entity_overlap),
    }

    return evaluation_results, global_metrics


# ==========================
# OPTIMIZARE ȘI FINE-TUNING
# ==========================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                  weight_decay=WEIGHT_DECAY)


def train_epoch_with_checkpoints(model, dataset, optimizer, device):
    model.train()
    total_loss = 0
    batch_losses = []

    for article_idx, article in enumerate(dataset):
        # Preprocesare articol (cod existent)
        windows = article.get("windows", [])
        summary = article.get("summary", "")

        input_ids_list = []
        attention_masks_list = []
        for window in windows:
            tokenized = tokenizer(
                window,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids_list.append(tokenized["input_ids"].squeeze(0))
            attention_masks_list.append(tokenized["attention_mask"].squeeze(0))

        input_ids = torch.cat(input_ids_list, dim=-1).to(device)
        attention_mask = torch.cat(attention_masks_list, dim=-1).to(device)

        input_ids = input_ids[: tokenizer.model_max_length].unsqueeze(0)
        attention_mask = attention_mask[: tokenizer.model_max_length].unsqueeze(
            0)

        labels = tokenizer(
            summary,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )["input_ids"].to(device)

        # Backpropagation
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())

        print(
            f"Article {article_idx+1}/{len(dataset)} done. Loss: {loss.item():.4f}", datetime.now())

    average_loss = total_loss / len(dataset)
    # Returnăm și numărul de articole procesate
    return average_loss, batch_losses


# Bucla principală pentru antrenare
for epoch in range(NUM_EPOCHS):
    print(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}", datetime.now())
    all_batch_losses = []
    # Apelăm funcția de antrenare pentru articolele rămase
    train_loss, batch_losses = train_epoch_with_checkpoints(
        model, train_data, optimizer, device
    )
    all_batch_losses.extend(batch_losses)
    print(
        f"Epoch {epoch + 1} completed. Training Loss: {train_loss:.4f}", datetime.now())

    torch.cuda.empty_cache()

    # Evaluăm modelul pe setul de validare
    results, global_metrics = evaluate_model(model, val_data, device)
    print(
        f"Validation metrics after epoch {epoch + 1}: {global_metrics}.\n", datetime.now())

    with open(f"validation_results_epoch_{epoch + 1}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generare grafic pentru Training Loss per batch/articol
    plt.figure()
    plt.plot(all_batch_losses, label="Training Loss")
    plt.xlabel("Batch/Article Index")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(f"training_loss_curve_{epoch+1}.png")
    plt.show()

# Salvăm modelul final după antrenare
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_epoch': NUM_EPOCHS,
}, "final_model.pth")

print("Training process completed. Model saved successfully after training.", datetime.now())
print("Starting final evaluation on test set...", datetime.now())

# Testare pe test.json
test_results, test_global_metrics = evaluate_model(model, test_data, device)

# Salvăm rezultatele testării
with open("test_results.json", "w") as f:
    json.dump(test_results, f, indent=4)

print("Test Results Saved.", datetime.now())

# Extragem metricile pentru grafice
rouge1_scores = [result["metrics"]["rouge"]["rouge1"]
                 for result in test_results]
rouge2_scores = [result["metrics"]["rouge"]["rouge2"]
                 for result in test_results]
rougeL_scores = [result["metrics"]["rouge"]["rougeL"]
                 for result in test_results]
bleu_scores = [result["metrics"]["bleu"] for result in test_results]

# Metricile NER detaliate
perfect_matches = [result["metrics"]["ner_detailed"]
                   ["perfect_matches"] for result in test_results]
partial_matches = [result["metrics"]["ner_detailed"]
                   ["partial_matches"] for result in test_results]
type_errors = [result["metrics"]["ner_detailed"]["type_errors"]
               for result in test_results]
spurious = [result["metrics"]["ner_detailed"]["spurious"]
            for result in test_results]
missing = [result["metrics"]["ner_detailed"]["missing"]
           for result in test_results]

# Metricile NER simple
simple_reference_entities = [
    result["metrics"]["ner_simple"]["reference_entities"] for result in test_results]
simple_summary_entities = [result["metrics"]["ner_simple"]
                           ["summary_entities"] for result in test_results]
simple_common_entities = [result["metrics"]["ner_simple"]
                          ["common_entities"] for result in test_results]
entity_overlap = [result["metrics"]["ner_simple"]
                  ["entity_overlap"] for result in test_results]

# Generare grafice
plt.figure()
plt.plot(rouge1_scores, label="ROUGE-1")
plt.plot(rouge2_scores, label="ROUGE-2")
plt.plot(rougeL_scores, label="ROUGE-L")
plt.plot(bleu_scores, label="BLEU")
plt.xlabel("Article Index")
plt.ylabel("Scores")
plt.title("Evaluation Metrics for Summaries")
plt.legend()
plt.savefig("evaluation_metrics.png")
plt.show()

plt.figure()
plt.plot(perfect_matches, label="Perfect Matches")
plt.plot(partial_matches, label="Partial Matches")
plt.plot(type_errors, label="Type Errors")
plt.plot(spurious, label="Spurious")
plt.plot(missing, label="Missing")
plt.xlabel("Article Index")
plt.ylabel("Counts")
plt.title("NER Detailed Evaluation Metrics")
plt.legend()
plt.savefig("ner_detailed_metrics.png")
plt.show()

plt.figure()
plt.plot(simple_reference_entities, label="Reference Entities")
plt.plot(simple_summary_entities, label="Summary Entities")
plt.plot(simple_common_entities, label="Common Entities")
plt.plot(entity_overlap, label="Entity Overlap")
plt.xlabel("Article Index")
plt.ylabel("Counts/Ratio")
plt.title("NER Simple Evaluation Metrics")
plt.legend()
plt.savefig("ner_simple_metrics.png")
plt.show()