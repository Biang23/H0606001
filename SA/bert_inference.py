# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-06-09 10:48:56

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer
import torch

path = "models/bert-base-cased-sst-2-lr==1e-5_max_sl==384"

model = AutoModelForSequenceClassification.from_pretrained(path)

text = "This comedy is not funny at all."

tokenizer = AutoTokenizer.from_pretrained(path)
tokenized_text = tokenizer(text, padding='max_length', max_length=384, truncation=True, return_tensors='pt')

with torch.no_grad():
    output = model(**tokenized_text)


logits = output.logits
probs = torch.softmax(logits, dim=-1)
predicted_label = torch.argmax(probs, dim=-1).item()
print(f"current input: {text}")
print(predicted_label)