import spacy
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize, FreqDist
from nltk.tokenize.toktok import ToktokTokenizer
import re
import string
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata
import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np

nlp = spacy.load('en', parse=True, tag=True, entity=True)
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    removed_text = soup.get_text()
    return removed_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text, mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(mapping.keys())),flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = mapping.get(match)\
                                if mapping.get(match)\
                                else mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match,text)
    expanded_text = re.sub("'","", expanded_text)
    return expanded_text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def stemmer(text):
    ps = nltk.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatization(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_punctuations(text, is_lower=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower:
        filtered_tokens = [token for token in tokens if token not in string.punctuation]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in string.punctuation]
    punc_filtered_text = ' '.join(filtered_tokens)
    return punc_filtered_text


def exploratory_data_analysis(text):
    eda_dict = {}
    # How many characters in the article
    length = len(text)
    eda_dict['length'] = length
    tokens = word_tokenize(text)
    tokens_nop = [t for t in tokens if t not in string.punctuation]
    # How many words in the article
    token_length = len(tokens)
    eda_dict['token_length'] = token_length
    eda_dict['token_length_without_punc'] = len(tokens_nop)
    # How many unique words
    unique = set(tokens)
    eda_dict['unique words count'] = len(unique)
    # Number of single characters
    single = [token for token in unique if len(token) == 1]
    eda_dict['single character count'] = len(single)
    # Frequency Distribution of words
    fd = nltk.FreqDist(tokens)
    eda_dict['frequency distribution'] = fd
    # Top 50 common words
    common = fd.most_common(50)
    # Frequency distribution plot of the 10 most common words
    # fd.plot(10)
    # How long are the words
    fd_wlen = nltk.FreqDist([len(w) for w in unique])
    eda_dict['long word count'] = list(filter(lambda x: x[1]>=1,fd_wlen.items()))
    # Bigrams and Trigrams
    bigr = nltk.bigrams(tokens[:10])
    eda_dict['bigrams'] = list(bigr)
    trigr = nltk.trigrams(tokens[:10])
    eda_dict['trigrams'] = list(trigr)
    return eda_dict


def wordcloud(text):
    wc = WordCloud(background_color="white").generate(text)
    return wc


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

        # lemmatize the text
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

        normalized_corpus.append(doc)

    return normalized_corpus













