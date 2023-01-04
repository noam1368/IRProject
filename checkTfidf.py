# set of documents
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict,Counter
import re
import nltk
import pickle
import numpy as np
nltk.download('stopwords')

from nltk.corpus import stopwords
from tqdm import tqdm
import operator
from itertools import islice,count
from contextlib import closing

import json
from io import StringIO
from pathlib import Path
from operator import itemgetter
import pickle
import matplotlib.pyplot as plt

#cossim
from sklearn.utils.fixes import sklearn


from collections import Counter, OrderedDict
import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
from google.cloud import storage
import math
from contextlib import closing
from inverted_index_gcp import MultiFileReader
import inverted_index_gcp
import pickle
from inverted_index_gcp import InvertedIndex
from traitlets.traitlets import Long


def get_docs_title_by_id(lst):
    all_titles = []
    for i,d in enumerate(lst):
        if d[0] in indexTitle:
            all_titles.append((d[0],docs_titles[d[0]]))

    return all_titles



def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in english_stopwords]
    return list_of_tokens

def posting_lists_reader(inverted_index, term): # work 2
    """  reads one posting list from disk in byte and convert to int
        return:
            [(doc_id:int, tf:int), ...].
    """
    with closing(MultiFileReader()) as reader:
        locs = inverted_index.posting_locs[term]
        b = reader.read(locs, inverted_index.df[term] * TUPLE_SIZE)# convert the bytes read into `b` to a proper posting list.
        posting_list = []

        while b:
            b_doc_id = int.from_bytes(b[0:4], "big")
            b_tf_of_w = int.from_bytes(b[4:6], "big")
            b = b[6:]
            posting_list.append((b_doc_id, b_tf_of_w))

        return posting_list

def get_top_n(lst, N = 5): #sort the list according to the x[1] and return the top N
    return sorted(lst, key= lambda x:x[1], reverse=True)[:N]

def query_get_top_N_tfidf(inverted_index, query_to_search, N = 5):
    """

    Args:
        query_to_search:
        index:

    Returns:

    """
    result = {}  # result[doc_id] = score
    epsilon = .0000001
    counter = Counter(query_to_search)
    DL_length = len(DL)
    query_length = len(query_to_search)

    for token in np.unique(query_to_search):
        if token in inverted_index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / query_length  #term frequency divded by the length of the query
            df = inverted_index.df[token]
            idf = math.log((DL_length) / (df + epsilon), 10)  #todo ass4 -> make save DL in memory We save a dictionary named DL, which fetches the document length of each document.
            docs = posting_lists_reader(inverted_index,token) #will return the list of docs, from byte to  [(doc_id:int, tf:int), ...]
            Q = tf*idf

            for doc_id, doc_tf in docs:
                D = (doc_tf / DL[doc_id]) * math.log(DL_length / inverted_index.df[token], 10)
                tfidf = D*Q
                result[doc_id] = result.get(doc_id, 0) + tfidf

    return get_top_n(result.items(),N)







###########################################

data = ['The sky is blue and we can see the blue sun.',
        'The sun is bright and yellow.',
        'here comes the blue sun',
        'Lucy in the sky with diamonds and you can see the sun in the sky',
        'sun sun blue sun here we come',
        'Lucy likes blue bright diamonds']

queries = ['look the the blue sky', 'He likes the blue the sun', 'Lucy likes blue sky with diamonds']


if __name__ == '__main__':
    for i in data:
        DL


    tokens = tokenize(query.lower())
    tokens_after_filter = [token for token in tokens if token in indexText.df[token] < 250000 and indexText.df] #todo change the number according to number we see warking good
    bestDocs = query_get_top_N_tfidf(indexText,tokens_after_filter,100)
    res = get_docs_title_by_id(bestDocs)