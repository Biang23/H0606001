# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-06-07 15:02:31

import os
import json
from fire import Fire

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd

def read_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def tfidf(dataset: pd.DataFrame, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    joint_sentence = [" ".join(d['tokenized_sentence']) for d in dataset]

    tfidf_matrix = vectorizer.fit_transform(joint_sentence)
    feature_name = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame()

    for idx, row in enumerate(tfidf_matrix):
        nonzero_indices = row.nonzero()[1]
        nonzero_tfidf = row.data
        tokens_tfidf = {
            feature_name[i]: nonzero_tfidf[j] for j, i in enumerate(nonzero_indices)
        }

        record = {
            'idx': dataset[idx]['idx'],
            'original_text': dataset[idx]['sentence'],
            'tokenized_text': " ".join(dataset[idx]['tokenized_sentence']),
            'tfidf': tokens_tfidf,
            'label': dataset[idx]['label']
        }

        tfidf_df = tfidf_df._append(record, ignore_index=True)
    
    return tfidf_df


def main(dataset_path: str):
    dataset = read_data(dataset_path)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_df = tfidf(dataset, vectorizer)
    tfidf_df.to_csv(f"{dataset_path.split('.')[0]}.csv", index=False)
    print("Done")

if __name__=="__main__":
    Fire(main)
