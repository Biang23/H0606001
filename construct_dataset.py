# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-06-07 16:50:34

import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from typing import Union, Dict
from collections import defaultdict
from fire import Fire
import logging
import sys
import os

import transformers
import datasets
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def _load_dataset(dataset_name: str) -> Union[pd.DataFrame, datasets.DatasetDict]:
    if dataset_name.endswith("csv"):
        return pd.read_csv(dataset_name)
    elif dataset_name.endswith("xlsx"):
        return pd.read_excel(dataset_name)
    elif dataset_name.endswith("parquet"):
        return pd.read_parquet(dataset_name)
    elif dataset_name.endswith("json"):
        return json.load(open(dataset_name, 'r', encoding='utf-8'))
    else:
        print("Loading huggingface dataset...")
        return load_dataset(dataset_name)


def tokenization(tokenizer: transformers.AutoTokenizer, dataset: Union[datasets.DatasetDict, pd.DataFrame]) -> Dict:

    examples = []
    tokenized_examples = defaultdict(list)

    if isinstance(dataset, datasets.DatasetDict):
        for split in dataset.keys():
            split_data = dataset[split]

            # for specific sst2 dataset, please modify the key to extract texts for different dataset
            for data in split_data:
                idx = data['idx']
                sentence = data['sentence']
                label = data['label']

                tokenized_sentence = tokenizer.tokenize(sentence)

                tokenized_examples[split].append({
                    'idx': idx,
                    'sentence': sentence,
                    'tokenized_sentence': tokenized_sentence,
                    'label': label
                })

    # For self-constructed dataset, assume we store the text in the 'text' field 
    # and the dataset has not been splited to train/validation/test datasets
    elif isinstance(dataset, pd.DataFrame): 
        
        for _, data in dataset.iterrows():
            idx = data['idx']
            text = data['text']
            label = data['label']

            tokenized_text = tokenizer.tokenize(text)

            examples.append({
                'idx': idx,
                'text': text,
                'tokenized_text': tokenized_text,
                'label': label
            })

        train, test = train_test_split(examples, test_size=0.2, shuffle=True, random_state=42)
        train, validation = train_test_split(train, test_size=0.25, shuffle=True, random_state=42)

        tokenized_examples['train'].extend(train)
        tokenized_examples['validation'].extend(validation)
        tokenized_examples['test'].extend(test)

    else:
        raise TypeError("The dataset must be datasets.DatasetDict or pandas.DataFrame")
    
    if isinstance(tokenized_examples, dict):
        return tokenized_examples
    else:
        raise TypeError(f"Unmatched return type as {type(tokenized_examples)}")


def save_dataset(tokenized_examples: dict, dataset_name: str):
    if not os.path.exists(f"data/{dataset_name}"):
        os.makedirs(f"data/{dataset_name}")
    for k, v in tokenized_examples.items():
        with open(f"data/{dataset_name}/{k}.json", 'w', encoding='utf-8') as f:
            json.dump(v, f, ensure_ascii=False, indent=4)
        f.close()
    logger.info(f"datasets saving done as {list(tokenized_examples.keys())}. ")
    

def main():

    model_name = "google-bert/bert-base-uncased"
    dataset_name = "stanfordnlp/sst2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sst2 =  _load_dataset(dataset_name)
    tokenized_examples = tokenization(tokenizer=tokenizer, dataset=sst2)
    save_dataset(tokenized_examples, dataset_name)


if __name__=="__main__":
    Fire(main)