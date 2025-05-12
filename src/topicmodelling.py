# Import necessary libraries
import os
import re
import string
import tldextract

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
from tqdm import tqdm

import collections

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk

from umap import UMAP
from hdbscan import HDBSCAN

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.metrics import pairwise_distances
from bertopic.representation import MaximalMarginalRelevance
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora

# add labels to dataframe, both for train and test set
def add_topics_to_df(df, topics, docs_sample):
    text_topics_dict = dict()
    for i,t in enumerate(topics):
        text_topics_dict[docs_sample[i]] = t
    print(text_topics_dict)
    text_topics_df = pd.DataFrame.from_dict(text_topics_dict, orient='index', columns=['topic']).reset_index().rename(columns={'index':'preprocessed_text'})

    new_df = pd.merge(left = df , right = text_topics_df, how='outer')

    return new_df



def create_vocabulary(docs_,stopwords):
    
    # Extract vocab to be used in BERTopic
    vocab = collections.Counter()
    tokenizer = CountVectorizer(stop_words = stopwords).build_tokenizer()
    for doc in tqdm(docs_):
        vocab.update(tokenizer(doc))
    vocab = [word for word, frequency in vocab.items() if frequency >= 100]
    return vocab