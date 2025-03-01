#!/usr/bin/env python3
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PegasusConfig,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    AdamW,
)
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

nltk.download("wordnet")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# SECȚIUNEA DE HIPERPARAMETRI
# ==========================

# Antrenare
BATCH_SIZE = 4
NUM_EPOCHS = 6
LEARNING_RATE = 5e-5

# Model
MODEL_NAME = "google/pegasus-xsum"
D_MODEL = 768
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
ENCODER_ATTENTION_HEADS = 12
DECODER_ATTENTION_HEADS = 12
FFN_DIM = 4 * D_MODEL
DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1
ACTIVATION_FUNCTION = "gelu"
MAX_POSITION_EMBEDDINGS = 512

# Generare
MAX_LENGTH = 512
NUM_BEAMS = 8
EARLY_STOPPING = True

# ==========================
# CONFIGURARE MODEL ȘI TOKENIZER
# ==========================
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
# tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# config = PegasusConfig(
#     vocab_size=len(tokenizer),
#     d_model=D_MODEL,
#     encoder_layers=ENCODER_LAYERS,
#     decoder_layers=DECODER_LAYERS,
#     encoder_attention_heads=ENCODER_ATTENTION_HEADS,
#     decoder_attention_heads=DECODER_ATTENTION_HEADS,
#     encoder_ffn_dim=FFN_DIM,
#     decoder_ffn_dim=FFN_DIM,
#     dropout=DROPOUT,
#     attention_dropout=ATTENTION_DROPOUT,
#     activation_function="gelu",
#     max_position_embeddings=MAX_LENGTH
# )
# model = PegasusForConditionalGeneration(config).to(device)
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(
    "cuda"
)
# ==========================
# ÎNCĂRCARE ȘI CONFIGURARE DATE
# ==========================
train_path = (
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/train.json"
)
val_path = (
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/val.json"
)
test_path = (
    "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/test.json"
)


def load_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


train_data = load_data(train_path)
val_data = load_data(val_path)
test_data = load_data(test_path)


# ==========================
# FUNCȚII AUXILIARE
# ==========================
def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(pred, ref) for pred, ref in zip(predictions, references)
    ]

    predictions_tokenized = [pred.split() for pred in predictions]
    references_tokenized = [ref.split() for ref in references]

    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], pred, smoothing_function=smoothie)
        for pred, ref in zip(predictions_tokenized, references_tokenized)
    ]

    return {
        "rouge": rouge_scores,
        "bleu": bleu_scores,
    }


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
    all_predictions = []
    all_references = []
    with torch.no_grad():
        for article in dataloader:
            article_id = article["id"]
            sections = article.get("windows", [])
            summary = article["summary"]

            summary_windows = []
            for i, window_text in enumerate(sections):
                window_summary = summarize_window(window_text)
                summary_windows.append(window_summary)

            final_summary = " ".join(summary_windows)
            all_predictions.append(final_summary)
            all_references.append(summary)

    metrics = compute_metrics(all_predictions, all_references)
    return metrics["rouge"], metrics["bleu"]


# ==========================
# OPTIMIZARE ȘI FINE-TUNING
# ==========================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


def train_epoch(model, dataset, optimizer, device):
    model.train()
    total_loss = 0

    for article_idx, article in enumerate(dataset):

        print(f"\n=== DEBUG: Training Article {article_idx + 1}/{len(dataset)} ===")
        print(f"Article ID: {article.get('id', 'Unknown')}")

        windows = article.get("windows", [])
        summary = article.get("summary", "")

        # Concatenăm input-urile
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

        # Pregătim intrările și etichetele
        input_ids = torch.cat(input_ids_list, dim=-1).to(device)
        attention_mask = torch.cat(attention_masks_list, dim=-1).to(device)

        input_ids = input_ids[: tokenizer.model_max_length].unsqueeze(0)  # Batch size 1
        attention_mask = attention_mask[: tokenizer.model_max_length].unsqueeze(0)

        labels = tokenizer(
            summary,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )["input_ids"].to(device)

        # Backpropagation
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Generăm rezumatul pentru debugging folosind funcția generate_summary_from_windows
        generated_summary = generate_summary_from_windows(windows)

        # Debugging pentru articol
        print(f"Article {article_idx + 1} Loss: {loss.item():.4f}")
        print(f"Generated Summary: {generated_summary[:300]}...")
        print(f"Reference Summary: {summary[:300]}...")
        print("======================================\n")

        total_loss += loss.item()

    # Calculăm pierderea medie
    average_loss = total_loss / len(dataset)
    print(f"Epoch completed. Average Training Loss: {average_loss:.4f}")
    return average_loss


if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_data, optimizer, device)
        print(f"Training loss after epoch {epoch + 1}: {train_loss:.4f}")
        rouge_scores, bleu_scores = evaluate_model(model, val_data, device)
        print(f"Validation ROUGE: {rouge_scores}")
        print(f"Validation BLEU: {bleu_scores}")

    print("Evaluating on test set...")
    rouge_scores, bleu_scores = evaluate_model(model, test_data, device)
    print(f"Test ROUGE: {rouge_scores}")
    print(f"Test BLEU: {bleu_scores}")
