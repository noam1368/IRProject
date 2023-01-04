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



data = ['The sky is blue and we can see the blue sun.',
        'The sun is bright and yellow.',
        'here comes the blue sun',
        'Lucy in the sky with diamonds and you can see the sun in the sky',
        'sun sun blue sun here we come',
        'Lucy likes blue bright diamonds']



RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
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
                      token.group() not in stopwords_frozen]
    return list_of_tokens


#
# def tf_idf_scores(data):
#     """
#     This function calculates the tfidf for each word in a single document utilizing TfidfVectorizer via sklearn.
#
#     Parameters:
#     -----------
#       data: list of strings.
#
#     Returns:
#     --------
#       Two objects as follows:
#                                 a) DataFrame, documents as rows (i.e., 0,1,2,3, etc'), terms as columns ('bird','bright', etc').
#                                 b) TfidfVectorizer object.
#
#     """
#     # YOUR CODE HERE
#     vectorizer = TfidfVectorizer(stop_words='english')  # object vector without the stop words
#     vectors = vectorizer.fit_transform(data)  # vector of (doc,term),tfidf
#     feature_names = vectorizer.get_feature_names_out()  # get the terms
#     dense = vectors.todense().tolist()
#     df = pd.DataFrame(dense, columns=feature_names)  # making a data frame
#     return df, vectorizer
#
#
# def cosine_sim_using_sklearn(queries, tfidf):
#     """
#     In this function you need to utilize the cosine_similarity function from sklearn.
#     You need to compute the similarity between the queries and the given documents.
#     This function will return a DataFrame in the following shape: (# of queries, # of documents).
#     Each value in the DataFrame will represent the cosine_similarity between given query and document.
#
#     Parameters:
#     -----------
#       queries: sparse matrix represent the queries after transformation of tfidfvectorizer.
#       documents: sparse matrix represent the documents.
#
#     Returns:
#     --------
#       DataFrame: This function will return a DataFrame in the following shape: (# of queries, # of documents).
#       Each value in the DataFrame will represent the cosine_similarity between given query and document.
#     """
#     return pd.DataFrame(cosine_similarity(queries, tfidf))
#
#
#
#
# if __name__ == '__main__':
#
#     #tf idf
#     df_tfidfvect, tfidfvectorizer = tf_idf_scores(data)
#     print("df_tfidfvect: ", df_tfidfvect)
#     print("tfidfvectorizer", tfidfvectorizer)
#
#     # tests
#     # assert df_tfidfvect.shape[1] == 10 and df_tfidfvect.shape[0] == 6
#     # assert 'is' not in df_tfidfvect.columns and 'we' not in df_tfidfvect.columns
#     # assert 'sun' in df_tfidfvect.columns and 'yellow' in df_tfidfvect.columns
#     # assert round(df_tfidfvect.max(), 3).max() == 0.798
#     # assert np.count_nonzero(df_tfidfvect) == 21
#     # assert type(tfidfvectorizer) == TfidfVectorizer
#
#     #vectorize this queries
#     #todo insert into function
#     queries = ['look the the blue sky', 'He likes the blue the sun', 'Lucy likes blue sky with diamonds']
#     queries_vector = tfidfvectorizer.transform(queries)
#
#     #cosine similarity
#
#     cosine_sim_df = cosine_sim_using_sklearn(queries_vector, df_tfidfvect)
#     print(cosine_sim_df)
#
#     # tests for cosine similarity
#     # assert cosine_sim_df.shape[0] == len(queries)
#     # assert cosine_sim_df.shape[1] == len(data)
#     # assert (abs(cosine_sim_df) > 1).any().any() == False
#     # assert np.count_nonzero(cosine_sim_df) == 16
#     # assert round(cosine_sim_df.max(), 3).max() == 0.888





def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / DL[str(doc_id)]) * math.log(len(DL) / index.df[term], 10)) for
                               doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates





def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE
    # print(Q)
    # print(D)
    cossim_dict = {}
    normelize_Q = np.linalg.norm(Q)

    for row_doc_id in D.iterrows():
        # cos_sim = dot(a, b)/(norm(a)*norm(b)) the formula i need to do each doc
        # = dot(Q, doc)/ norm(Q) * norm(doc)
        b = normelize_Q * np.linalg.norm(row_doc_id[1])
        cos_sim = np.dot(Q, row_doc_id[1]) / b

        cossim_dict[row_doc_id[0]] = cos_sim

    return cossim_dict





def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1], reverse = True)[: N]

def get_topN_score_for_query(query,index,N=3):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    key: query_id
    value: list of pairs in the following format:(doc_id, score). """

    """cosine_similarity(D,Q) --> we will take the top N of the scores """
    query_cosine_similarity_to_docs = cosine_similarity( generate_document_tfidf_matrix(queries_to_search[key],index,words,pls) ,
                                  generate_query_tfidf_vector(queries_to_search[key],index))

    return get_top_n(query_cosine_similarity_to_docs,N)



