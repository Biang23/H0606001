# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-06-08 17:06:04

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from collections import defaultdict
from fire import Fire

def read_data(data_file: str):
    return pd.read_csv(data_file)

def data_processing(dataset: pd.DataFrame) -> dict:
    data = defaultdict(float)
    for _, d in dataset.iterrows():
        for k, v in eval(d['tfidf']).items():
            data[k] += v
    return data


def _wordcloud(data: dict, save_path: str):

    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate_from_frequencies(data)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.savefig(save_path)
    plt.show()

    return wordcloud

def main(data_file, save_path):
    dataset = read_data(data_file)
    data = data_processing(dataset)
    _wordcloud(data, save_path)

if __name__=="__main__":
    Fire(main)