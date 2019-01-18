import pandas as pd
import os
from Clustering.Preprocessing import *
import pprint
import spacy


def load_data():
    path = "C:/Users/madfa/Downloads/Git Repository/Document Dictionary/Documents.csv"
    data = pd.read_csv(path,encoding="utf-8")
    return data


class textMining:
    news_df = load_data()

    # Combining Headlines and Content
    news_df['full_text'] = news_df['Headlines'].map(str) + '. ' + news_df['Content']

    # pre-process the text and store
    news_df['normalized_text'] = normalize_corpus(news_df['full_text'])
    norm_corpus = list(news_df['normalized_text'])

    # write the pre-processed text to a new csv
    news_df.to_csv('pre-processed_news.csv', index=False, encoding='utf-8')

    # exploratory data analysis
    eda = exploratory_data_analysis(news_df['Content'][1])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(eda)

    # nlp_corpus = normalize_corpus(news_df['full_text'], text_lower=False,
    #                               lemmatization=False, special_char_removal=False)
    # # POS tagging with spacy
    # spacy_pos_tagged = [(word, word.tag_, word.pos_) for word in ]





