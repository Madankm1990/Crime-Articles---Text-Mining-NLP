import pandas as pd
import os
from Clustering.Preprocessing import *
import pprint

def load_data():
    path = "C:/Users/madfa/Downloads/Git Repository/Document Dictionary/Documents.csv"
    data = pd.read_csv(path,encoding="utf-8")
    return data

def normalize_corpus(corpus, html_stripping=True, contraction = True,
                     accented_char_removal = True, text_lower = True,
                     lemmatization = True, special_char_removal = True,
                     stopword_removal = True, remove_digits = False):
    normalized_corpus = []

    for doc in corpus:
        # strip HTML tags
        if html_stripping:
            doc = remove_html_tags(doc)

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # expand contractions(doc)
        if contraction:
            doc = expand_contractions(doc)

        # convert to lower case
        if text_lower:
            doc = doc.lower()

        # remove extra new lines
        doc = re.sub(r'[\r|\n|\r\n]+',' ', doc)

        # lemmatize text
        if lemmatization:
            doc = lemmatization(doc)

        # remove special characters
        if special_char_removal:
            special_char = re.compile(r'({.(-)!}])')
            doc = special_char.sub(" \\1", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower=text_lower)

class ImportData:
  news_df = load_data()
  eda = exploratory_data_analysis(news_df['Content'][1])
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(eda)





