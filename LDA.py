# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-06-08 15:59:32

import pandas as pd
from fire import Fire

from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

from tfidf import read_data


def main(data_file):
    dataset = read_data(data_file)

    tokenized_sentences = [" ".join(d['tokenized_sentence']) for d in dataset]

    # 使用CountVectorizer进行词频矩阵构建，避开停用词
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(tokenized_sentences)

    # 使用LDA进行主题建模
    lda = LatentDirichletAllocation(n_components=10, random_state=42)  # 假设我们想要10个主题
    lda.fit(count_matrix)

    # 输出每个主题的前10个词语
    n_top_words = 10
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__=="__main__":
    Fire(main)