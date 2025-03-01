import os
import json
import logging
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    AutoTokenizer
)
from trl import PPOConfig, PPOTrainer
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

# Configurare Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Dezactivează Weights & Biases
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

# Configurare Căi Către Date
train_path = "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/train_windows.json"
val_path = "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/val_windows.json"
test_path = "/export/home/mecanica/stud/m/matei.neaga/NN/proiect/prelucrare_date/test_windows.json"

# Funcție pentru Încărcarea Datelor
def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise e

# Funcție pentru Conversia Datelor în Dicționar
def convert_to_dict(data):
    keys = data[0].keys()
    for d in data:
        if not all(key in d for key in keys):
            raise ValueError(f"Inconsistent data structure in {d}")
    return {key: [d[key] for d in data] for key in keys}

# Încărcare Date
logger.info("Loading datasets...")
train_data = load_data(train_path)[:4227]  # Limitează setul de date pentru testare
val_data = load_data(val_path)[:5]
test_data = load_data(test_path)[:5]

train_dict = convert_to_dict(train_data)
val_dict = convert_to_dict(val_data)
test_dict = convert_to_dict(test_data)

train_dataset = Dataset.from_dict(train_dict)
val_dataset = Dataset.from_dict(val_dict)
test_dataset = Dataset.from_dict(test_dict)

# Configurare Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Funcție pentru Preprocesarea Datelor
def prepare_dataset(dataset, tokenizer):
    """Pre-tokenize the dataset before training."""
    def tokenize(element):
        windows = element["windows"]
        if any(isinstance(item, list) for item in windows):
            windows = [subitem for item in windows for subitem in (item if isinstance(item, list) else [item])]
        text = "".join(windows).strip()
        if not text:  # Verifică dacă textul este gol
            return {"input_ids": []}  # Returnează o listă goală pentru texte invalide
        outputs = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=1024
        )
        if len(outputs["input_ids"]) == 0:
            return {"input_ids": []}  # Returnează o listă goală pentru texte invalide
        return {"input_ids": outputs["input_ids"]}

    # Aplică tokenizarea
    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names,
        num_proc=None  # Dezactivează multiprocessing-ul pentru a preveni OOM
    )

    # Filtrează exemplele goale
    logger.info(f"Number of examples before filtering: {len(dataset)}")
    dataset = dataset.filter(lambda x: isinstance(x["input_ids"], list) and len(x["input_ids"]) > 0)
    logger.info(f"Number of examples after filtering: {len(dataset)}")

    return dataset

# Preprocesare Dataset
logger.info("Preprocessing datasets...")
train_dataset = prepare_dataset(train_dataset, tokenizer)
val_dataset = prepare_dataset(val_dataset, tokenizer)
test_dataset = prepare_dataset(test_dataset, tokenizer)

# Verifică dacă dataset-ul este gol
if len(train_dataset) == 0:
    raise ValueError("Train dataset is empty after filtering!")
if len(val_dataset) == 0:
    raise ValueError("Validation dataset is empty after filtering!")
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after filtering!")

logger.info(f"Train dataset shape: {train_dataset.shape}")
logger.info(f"First example in train dataset: {train_dataset[0]}")

# Configurare Device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Încărcare Modele
logger.info("Loading models...")
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model_saved", num_labels=1).to(device)
value_model = AutoModelForSequenceClassification.from_pretrained("reward_model_saved", num_labels=1).to(device)

# Încărcare Model Pegasus
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
ref_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)

# Configurare Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Configurare TensorBoard
log_dir = "./RL_logs"
writer = SummaryWriter(log_dir)

# Configurare PPOTrainer
logger.info("Initializing PPO Trainer...")
training_args = PPOConfig(
    output_dir="./ppo_trainer",
    batch_size=8,
    mini_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    log_with="tensorboard",
)

trainer = PPOTrainer(
    args=training_args,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizer=optimizer,
)

# Antrenare
logger.info("Starting training...")
num_epochs = 3
for epoch in range(num_epochs):
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
    val_loss = 0.0  # Placeholder pentru val_loss
    logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Validation loss: {val_loss:.4f}")

# Salvare Model Final
logger.info("Saving final model...")
trainer.save_model("final_trained_model")
writer.flush()
writer.close()
logger.info("Training completed successfully!")