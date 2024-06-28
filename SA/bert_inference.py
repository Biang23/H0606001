# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-06-09 10:48:56

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer
import torch

sentiments = {
    '0': 'negative',
    '1': 'positive',
}

def sentiment_analysis(model_path):
    while True:
        text = input("Please input your film content.")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenized_text = tokenizer(text, padding='max_length', max_length=384, truncation=True, return_tensors='pt')

        with torch.no_grad():
            output = model(**tokenized_text)

        logits = output.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        predictions = sentiments[predicted_label]
        
        print(f"current input: {text}")
        print(f"cuurent_prediction: {predictions}")

def main():

    path = "models/bert-base-cased-sst-2-lr==1e-5_max_sl==384"

    sentiment_analysis(path)

if __name__=="__main__":
    import fire

    fire.Fire(main)
