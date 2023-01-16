from collections import Counter
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from google.cloud import storage
import math
from contextlib import closing
import pickle
from inverted_index_gcp import *
from inverted_index_gcp import MultiFileReader

###################
# making local variable
bucket_name = 'new_index_bucket'
client = storage.Client()
bucket = client.bucket(bucket_name)

# TEXT
index_src = "indexTexts.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
indexText = pickle.loads(pickel_in)

# Title
index_src = "indexTitles.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
indexTitle = pickle.loads(pickel_in)

# pageview
index_src = "page_view.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
page_views = pickle.loads(pickel_in)

# DL
index_src = "dl.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
DL = pickle.loads(pickel_in)

# page_rank
index_src = "page_rank.pickle"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
page_ranks = pickle.loads(pickel_in)

# id_titles
index_src = "id_titles.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
titles = pickle.loads(pickel_in)

# id_titles
index_src = "nf.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
NF = pickle.loads(pickel_in)

# Anchor
index_src = "indexAnchors.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
indexAnchor = pickle.loads(pickel_in)

# prefix for the local folder inside the instance were the posting locs bin files are found
text_prefix = "poasting_locs_new/posting_text/"
title_prefix = "poasting_locs_new/posting_title/"
anchor_prefix = "poasting_locs_new/posting_anchor/"

# BM25 parameter that calculate one time for improve the run time
length_docLengths = len(DL)  # to run faster one time calculate
dl_list = list(DL.items())
dl_list = list(map(lambda x: int(x[1]), dl_list))
sum_dl = 0
for n in dl_list:
    sum_dl += n

avg_dl = sum_dl / length_docLengths

###############################################################################   all the things for tokenize
TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)  # this is containing all the stop words
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


############################################################################ id title dictionary to get the titles

def get_docs_title_by_id(lst):
    all_titles = []
    for id, no in lst:
        if id in titles:
            all_titles.append((id, titles[id]))

    return all_titles


############################################################################### Tokenize

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
                      token.group() not in all_stopwords]
    return list_of_tokens


###############################################################################


###############################################################################   get top N after Sort
def get_top_n(lst, N=5):  # sort the list according to the x[1] and return the top N
    return sorted(lst, key=lambda x: x[1], reverse=True)[:N]


###############################################################################   END

###############################################################################   reader for the posting list
def posting_lists_reader(inverted_index, term, prefix):
    """ A generator that reads one posting list from disk and yields
        a (word:str, [(doc_id:int, tf:int), ...]) tuple.
    """
    with closing(MultiFileReader()) as reader:
        locs = inverted_index.posting_locs[term]
        locs = [(prefix + v[0], v[1]) for v in locs]
        b = reader.read(locs, inverted_index.df[term] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted_index.df[term]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))

    return posting_list


###############################################################################   END
###############################################################################   BM25


class BM25:
    """
    class that represent all what the BM25 need for calculation
    """

    def __init__(self, inverted_index, k1, k3, b):
        self.inverted_index = inverted_index
        self.doc_lengths = DL
        self.N = length_docLengths
        self.avgdl = avg_dl
        self.b = b
        self.k1 = k1
        self.k3 = k3

    def get_scores(self, tokens):
        """
        Args:
            tokens: query after tokenize

        Returns:
            dictionary of scores according to bm25 according to the weight the constructor got as feilds k1,k3,b

        """
        query_Counter = Counter(tokens)
        print(query_Counter)
        scores = {}
        for term in tokens:
            if term not in self.inverted_index.df.keys():
                continue

            idf = math.log10((self.N + 1) / self.inverted_index.df[term])
            posting = posting_lists_reader(self.inverted_index, term, text_prefix)

            for doc_id, tf in posting:
                d = self.doc_lengths[doc_id]
                tf_ij = ((self.k1 + 1) * tf) / ((1 - self.b + self.b * d / self.avgdl) * self.k1 + tf)
                tf_iq = ((self.k3 + 1) * query_Counter[term]) / (self.k3 + query_Counter[term])
                score = tf_ij * idf * tf_iq
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += score
        return scores


def bm25(inverted_index, tokens, k1, k3, b, N):
    """
    use the BM25 object to get the bm25 scores

    Returns:
        sorted array of the top N (doc id,score) according to score
    """
    b = b
    bm = BM25(inverted_index, k1, k3, b)
    result = bm.get_scores(tokens)
    return get_top_n(list(result.items()), N)


###############################################################################   END
###############################################################################   COS SIM Body

def query_get_top_N_tfidf(inverted_index, query_to_search, N=5):
    """
    this method will calculate the cosine similarity of the query and docs,
    with the inverted index, posting of the term, normelize dictionary of the docs,
    and another feilds that has been save a head, to make the run time faster.

    Args:
        inverted_index: the index we want (we will use here just with index Body)
        query_to_search: tokens of the query
        N: the number of documents for the retrieval

    Returns:
        sorted array of the top N (doc id,score) according to score

    """
    result = {}  # result[doc_id] = score
    epsilon = .0000001  # woun't be really in use. but it is for the scenario of devide zero.
    query_length = len(query_to_search)
    tokens_counter = Counter(query_to_search)
    NF_length = len(NF)

    for token in np.unique(query_to_search):
        if token in inverted_index.df.keys():
            df = inverted_index.df[token]
            tf = tokens_counter[token] / query_length
            idf = math.log10((NF_length) / (df + epsilon))
            tfidfQ = tf * idf
            docs = posting_lists_reader(inverted_index, token, text_prefix)
            for doc_id, doc_tf in docs:
                tfidfD = (doc_tf / NF[doc_id]) * math.log10(NF_length / inverted_index.df[token])
                tfidf_mul = tfidfD * tfidfQ
                result[doc_id] = result.get(doc_id, 0) + tfidf_mul

    return get_top_n(result.items(), N)


###############################################################################   END

###############################################################################   Binary Score for Titles
def query_get_for_all_binary_Title(inverted_index, query_to_search, N=0):
    """
    this function will use binary search to give score of the tokens in the different titles.
    +1 for each occasion of a token/term from the query.

    Args:
        inverted_index: the index we want (we will use here just with index Body)
        query_to_search: tokens of the query
        N: the number of documents for the retrieval

    Returns:
        sorted array of the top N (doc id,score) according to score

    """
    result = {}
    for term in query_to_search:
        if inverted_index.df.get(term):
            ls_doc_freq = posting_lists_reader(inverted_index, term, title_prefix)
            for doc, freq in ls_doc_freq:
                result[doc] = result.get(doc, 0) + 1
    lst_doc = Counter(result).most_common()

    if N > 0:  # this is for the search method. we don't need all the titles
        return lst_doc[:N]

    return lst_doc


###############################################################################   END

###############################################################################   Binary Score for Anchor
def query_get_tfidf_for_all_Anchor(inverted_index, query_to_search, N=0):
    """
    this function will use binary search to give score of the tokens in the different anchors.
    +1 for each occasion of a token/term from the query.

    Args:
        inverted_index: the index we want (we will use here just with index Body)
        query_to_search: tokens of the query
        N: the number of documents for the retrieval

    Returns:
        sorted array of the top N (doc id,score) according to score

    """
    result = {}
    for term in query_to_search:
        if inverted_index.df.get(term):
            print(term)
            ls_doc_freq = posting_lists_reader(inverted_index, term, anchor_prefix)
            for doc, freq in ls_doc_freq:
                result[doc] = result.get(doc, 0) + 1

    lst_doc = Counter(result).most_common()

    if N > 0:  # this is for the search method. we don't need all the anchors
        return lst_doc[:N]

    return lst_doc


###############################################################################   END

###############################################################################   Merge function for the main search
def search_merge_results_with_page_rank(title_scores, body_scores, title_weight=0.5, text_weight=0.5,weight_page_rank= 1.5 ,N=10):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a list of tuples build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: score

    body_scores: a list of tuples build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: score
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 100, for the topN function.

    Returns:
    -----------
    lst of querires and topN pairs as follows:
                                            key: query_id
                                            value: list of pairs in the following format:(doc_id,score).
    """
    result = {}

    for k, v in title_scores:
        if k not in result:
            result[k] = 0

        result[k] = result[k] + (title_weight * v)

        if k in page_ranks:
             result[k] += (page_ranks[k] * weight_page_rank)

    for k, v in body_scores:
        if k not in result:
            result[k] = 0

        result[k] = result[k] + (text_weight * v)

        if k in page_ranks:
             result[k] += (page_ranks[k] * weight_page_rank)

    return sorted(result.items(), key=lambda x: x[1], reverse=True)[:N]


###############################################################################   END

###############################################################################   Flask
from flask import Flask, request, jsonify


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


###############################################################################   END

###############################################################################   Main methods
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    print("search")
    N = 20  # the number of docs i want to do the search on
    tokens = tokenize(query.lower())   
    bestDocsBody = bm25(indexText, tokens, 1.5, 1.5, 0.75, N)
    bestDocsTitle = query_get_for_all_binary_Title(indexTitle, tokens, N)
    bestDocs = search_merge_results_with_page_rank(bestDocsTitle, bestDocsBody, 0.5, 0.5, 1.5, N)
    res = get_docs_title_by_id(bestDocs)
    print(res)
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query.lower())
    bestDocs = query_get_top_N_tfidf(indexText, tokens, 100)
    res = get_docs_title_by_id(bestDocs)
    print(res[:10])
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = tokenize(query)
    bestDocs = query_get_for_all_binary_Title(indexTitle, tokens)
    res = get_docs_title_by_id(bestDocs)

    print(res[:10])
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = tokenize(query.lower())
    bestDocs = query_get_tfidf_for_all_Anchor(indexAnchor, tokens)
    res = get_docs_title_by_id(bestDocs)
    print(res[:10])
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    for id in wiki_ids:
        res.append(page_ranks.get(id, 0))
    print(res[:10])
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    for id in wiki_ids:
        res.append(page_views.get(id, 0))
    print(res[:10])
    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

########################################################################################   Expirement methods that we didn't use
"""
1. the first function is check for scoring the body with manipulate binary -> like the same 
    binary but the sum up is with the frequency of the terms and not +1.
"""
"""
2. the second function is check search func with including of page rank on the weights.
    got bad result so didn'y use.
"""

# def query_get_for_all_freq_text(inverted_index, query_to_search,N=0):
#     """
#     Args:
#         query_to_search:
#         index:
#
#     Returns:
#
#     """
#     result = {}
#     for term in query_to_search:
#         if inverted_index.df.get(term):
#             ls_doc_freq = posting_lists_reader(inverted_index, term, text_prefix)
#             for doc, freq in ls_doc_freq:
#                 result[doc] = result.get(doc, 0) + freq
#     lst_doc = Counter(result).most_common()
#
#     if N > 0: #this is for the search method. we don't need all the titles
#         return lst_doc[:N]
#
#     return lst_doc
#
#
#
# @app.route("/search_freq")
# def search_freq():
#     '''
#         relate : here we will take the regular binary search and improve it by
#         taking the freq of each term in each doc.
#         will give more accurate when using with text.
#
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     '''
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#     print("search_expansion")
#     N = 100 #the number of docs i want to do the search on
#     tokens = tokenize(query.lower())
#     bestDocs = query_get_for_all_freq_text(indexText,tokens,100)
#     res = get_docs_title_by_id(bestDocs)
#     print(res)
#     # END SOLUTION
#     return jsonify(res)


#
# def search_merge_results_with_page_rank(title_scores, body_scores, title_weight=0.3, text_weight=0.3, page_rank_weight = 0.4, N=100):
#     """
#     This function merge and sort documents retrieved by its weighte score (e.g., title and body).
#
#     Parameters:
#     -----------
#     title_scores: a list of tuples build upon the title index of queries and tuples representing scores as follows:
#                                                                             key: query_id
#                                                                             value: score
#
#     body_scores: a list of tuples build upon the body/text index of queries and tuples representing scores as follows:
#                                                                             key: query_id
#                                                                             value: score
#     title_weight: float, for weigted average utilizing title and body scores
#     text_weight: float, for weigted average utilizing title and body scores
#     N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 100, for the topN function.
#
#     Returns:
#     -----------
#     lst of querires and topN pairs as follows:
#                                             key: query_id
#                                             value: list of pairs in the following format:(doc_id,score).
#     """
#     page_rank_used = set()
#     result = {}
#
#     for k, v in title_scores:
#         if k not in result:
#             result[k] = 0
#
#         result[k] = result[k] + (title_weight * v)
#
#         if k not in page_rank_used:
#             result[k] += (page_ranks[k] * page_rank_weight)
#
#     for k, v in body_scores:
#         if k not in result:
#             result[k] = 0
#
#         result[k] = result[k] + (text_weight * v)
#
#         if k not in page_rank_used:
#             result[k] += (page_ranks[k] * page_rank_weight)
#
#     return sorted(result.items(), key=lambda x: x[1], reverse=True)[:N]


#
# @app.route("/search_with_page_rank")
# def search_with_page_rank():
#     ''' Returns up to a 100 of your best search results for the query. This is
#         the place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     '''
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#     print("search")
#     N = 100 #the number of docs i want to do the search on
#     tokens = tokenize(query.lower())
#     bestDocsBody = bm25(indexText,tokens,0.5,0.5,0.5,N)
#     bestDocsTitle = query_get_for_all_binary_Title(indexTitle, tokens,N)
#     bestDocs = search_merge_results_with_page_rank(bestDocsTitle,bestDocsBody,0.3,0.3,0.4,100)
#     res = get_docs_title_by_id(bestDocs)
#     print(res)
#     # END SOLUTION
#     return jsonify(res)
"""
def search_merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5 ,N=10):
 
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a list of tuples build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: score

    body_scores: a list of tuples build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: score
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 100, for the topN function.

    Returns:
    -----------
    lst of querires and topN pairs as follows:
                                            key: query_id
                                            value: list of pairs in the following format:(doc_id,score).

    result = {}

    for k, v in title_scores:
        if k not in result:
            result[k] = 0

        result[k] = result[k] + (title_weight * v)

    for k, v in body_scores:
        if k not in result:
            result[k] = 0

        result[k] = result[k] + (text_weight * v)

    return sorted(result.items(), key=lambda x: x[1], reverse=True)[:N]



@app.route("/search_test1")
def search_test1():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    print("search")
    N = 100  # the number of docs i want to do the search on
    tokens = tokenize(query.lower())
    length_tokens = len(tokens)
    if (length_tokens == 1):
        bestDocsTitle = query_get_for_all_binary_Title(indexTitle, tokens, N)
        res = get_docs_title_by_id(bestDocsTitle)
        print(res)
        return jsonify(res)

    else:
        bestDocsBody = bm25(indexText, tokens, 1, 0.5, 0.5, N)
        bestDocsTitle = query_get_for_all_binary_Title(indexTitle, tokens, N)
        bestDocs = search_merge_results(bestDocsTitle, bestDocsBody, 0.5, 0.5, N)
        res = get_docs_title_by_id(bestDocs)
        print(res)
    return jsonify(res)
"""

