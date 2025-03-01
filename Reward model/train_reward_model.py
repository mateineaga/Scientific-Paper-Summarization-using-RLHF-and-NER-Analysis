#!/usr/bin/env python3
import random
import pandas as pd
from operator import itemgetter
import torch
import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from trl import RewardTrainer, RewardConfig
import json


def load_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

device="cuda:0"

prepared_dataset = Dataset.from_list(load_data("merged.json"))
prepared_dataset.to_pandas()

print(prepared_dataset)

#Select a base model whch we need to train for reward modeling.
model_name = "distilroberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.max_length = 512
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


def formatting_func(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    prompt_plus_chosen_response = "".join(examples["windows"]) + "\n" + examples["summary"]
    if examples["generated-summary-long"] == None:
        prompt_plus_rejected_response = "".join(examples["windows"]) + "\n" + " "
    else:
        prompt_plus_rejected_response = "".join(examples["windows"]) + "\n" + examples["generated-summary-long"]
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }


formatted_dataset = prepared_dataset.map(formatting_func)
formatted_dataset = formatted_dataset.train_test_split()

# class MyTrainingArguments(TrainingArguments):
#     def __init__(self, *args, max_length=512, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.max_length = max_length

# Configuring the training arguments
training_args = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    logging_steps=1,
    num_train_epochs = 10,
    report_to=None
)
# Loading the RewardTrainer from TRL
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
)
trainer.train()
trainer.save_model("reward_model_saved")

