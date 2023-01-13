import gzip
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
from inverted_index_gcp import *
from inverted_index_gcp import MultiFileReader
# import inverted_index_gcp
import pickle
# from inverted_index_gcp import InvertedIndex
# from traitlets.traitlets import Long
###################

#work3
TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords) #this is containing all the stop words
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

#making local variable
bucket_name = 'new_index_bucket'
client = storage.Client()
bucket = client.bucket(bucket_name)

#TEXT
index_src = "indexTexts.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
indexText = pickle.loads(pickel_in)

#Title
index_src = "indexTitles.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
indexTitle = pickle.loads(pickel_in)

# #Anchor todo
# index_src = "indexAnchors.pkl"
# blob_index = bucket.blob(f"{index_src}")
# pickel_in = blob_index.download_as_string()
# indexAnchor = pickle.loads(pickel_in)

#pageview
index_src = "page_view.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
page_views = pickle.loads(pickel_in)

#DL
index_src = "dl.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
DL = pickle.loads(pickel_in)

#page_rank
index_src = "page_rank.pickle"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
page_ranks = pickle.loads(pickel_in)

#id_titles
index_src = "id_titles.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
titles = pickle.loads(pickel_in)

#id_titles
index_src = "norm.pkl"
blob_index = bucket.blob(f"{index_src}")
pickel_in = blob_index.download_as_string()
NF = pickle.loads(pickel_in)

#prefix for the local folder inside the instance were the posting locs bin files are found
text_prefix = "poasting_locs_new/posting_text/"
# text_prefix = "/content/posting_text/" #from colab
title_prefix = "poasting_locs_new/posting_title/"
# title_prefix = "/content/posting_title/"
anchor_prefix = "poasting_locs_new/posting_anchor/"

#BM25 parameter that calculate one time for improve the run time
length_docLengths = len(DL) #to run faster one time calculate
dl_list =  list(DL.items())
dl_list =list(map(lambda x: int(x[1]),dl_list))
sum_dl = 0
for n in dl_list:
  sum_dl += n

avg_dl = sum_dl / length_docLengths
############################################################################

def get_docs_title_by_id(lst):
    all_titles = []
    for id,no in lst:
        if id in titles:
            all_titles.append((id,titles[id]))
            # print((id,titles[id]))

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
                      token.group() not in all_stopwords]
    return list_of_tokens




###############################################################################   get top N after Sort
def get_top_n(lst, N = 5): #sort the list according to the x[1] and return the top N
    return sorted(lst, key= lambda x:x[1], reverse=True)[:N]
###############################################################################   END


###############################################################################   reader for the posting list
def posting_lists_reader(inverted_index, term ,prefix):
    """ A generator that reads one posting list from disk and yields
        a (word:str, [(doc_id:int, tf:int), ...]) tuple.
    """
    with closing(MultiFileReader()) as reader:
        locs = inverted_index.posting_locs[term]
        locs = [(prefix + v[0], v[1]) for v in locs]
        b = reader.read(locs, inverted_index.df[term] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted_index.df[term]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))

    return posting_list

###############################################################################   END
###############################################################################   BM25



class BM25:
    def __init__(self, inverted_index,k1,k3,b):
        self.inverted_index = inverted_index
        self.doc_lengths = DL
        self.N = length_docLengths
        self.avgdl = avg_dl
        self.b = b
        self.k1 = k1
        self.k3 = k3

    def get_scores(self, tokens):
            query_Counter = Counter(tokens)
            print(query_Counter)
            scores = {}
            for term in tokens:
                if term not in self.inverted_index.df.keys():
                    continue

                idf = math.log10((self.N+1)/self.inverted_index.df[term])
                posting = posting_lists_reader(self.inverted_index, term, text_prefix)

                for doc_id, tf in posting:
                  d = self.doc_lengths[doc_id]
                  tf_ij = ((self.k1+1)*tf)/((1 - self.b + self.b * d / self.avgdl)*self.k1+tf)
                  tf_iq = ((self.k3+1)*query_Counter[term])/(self.k3+query_Counter[term])
                  score =  tf_ij* idf  * tf_iq
                  if doc_id not in scores:
                      scores[doc_id] = 0
                  scores[doc_id] += score
            return scores


def bm25(inverted_index,tokens,k1,k3,b,N):
    b = b
    bm = BM25(inverted_index,k1,k3,b)
    result = bm.get_scores(tokens)
    return get_top_n(list(result.items()),N)

###############################################################################   END
###############################################################################   COS SIM Body

def query_get_top_N_tfidf(inverted_index, query_to_search, N=5):
    """

    Args:
        query_to_search:
        index:

    Returns:

    """

    # for embedding in query_to_search:
    #   print(embedding)

    result = {}  # result[doc_id] = score
    epsilon = .0000001
    counter = Counter(query_to_search)
    NF_length = len(NF)
    query_length = len(query_to_search)

    for token in np.unique(query_to_search):
        if token in inverted_index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / query_length  # term frequency divded by the length of the query
            df = inverted_index.df[token]
            idf = math.log10((NF_length) / (df + epsilon))
            docs = posting_lists_reader(inverted_index,token,text_prefix)  # will return the list of docs, from byte to  [(doc_id:int, tf:int), ...]
            tfidfQ = tf * idf
            for doc_id, doc_tf in docs:
                tfidfD = (doc_tf / NF[doc_id]) * math.log10(NF_length / inverted_index.df[token])
                tfidf_mul = tfidfD * tfidfQ
                result[doc_id] = result.get(doc_id, 0) + tfidf_mul

    return get_top_n(result.items(), N)



# def query_get_top_N_tfidf(inverted_index, query_to_search, N = 5): #todo delete another try for the func
#     """
#
#     Args:
#         query_to_search:
#         index:
#
#     Returns:
#
#     """
#     result = {}  # result[doc_id] = score
#     epsilon = .0000001
#     counter = Counter(query_to_search)
#     # query_length = len(query_to_search)
#     query_vec_2 = 0
#     for w in np.unique(query_to_search):
#         query_vec_2 += counter[w]**2
#     print(query_to_search)
#
#     for token in np.unique(query_to_search):
#         # print(token)
#         if token in inverted_index.df.keys():  # avoid terms that do not appear in the index.
#             docs = posting_lists_reader(inverted_index, token,text_prefix)  # will return the list of docs, from byte to  [(doc_id:int, tf:int), ...]
#             # print("Token: " ,token,"posting: ",docs)
#             # print(len(docs))
#             # print("inside if ", token)
#             for doc_id, doc_tf_w in docs:
#                 simCurrent = counter[token]*doc_tf_w
#                 result[doc_id] = result.get(doc_id, 0) + simCurrent
#                 result[doc_id] = result[doc_id] * (1 / ((query_vec_2 * NF[doc_id]) + epsilon))
#
#     return get_top_n(list(result.items()),N)

###############################################################################   END



def query_get_tfidf_for_all_Title(inverted_index, query_to_search,N=0):
    """
    Args:
        query_to_search:
        index:

    Returns:

    """
    result = {}
    for term in query_to_search:
        if inverted_index.df.get(term):
            ls_doc_freq = posting_lists_reader(inverted_index, term, title_prefix)
            for doc, freq in ls_doc_freq:
                result[doc] = result.get(doc, 0) + 1
    lst_doc = Counter(result).most_common()

    if N > 0: #this is for the search method. we don't need all the titles
        return lst_doc[:N]

    return lst_doc



def query_get_tfidf_for_all_Anchor(inverted_index, query_to_search):
    """
    Args:
        query_to_search:
        index:

    Returns:

    """
    result = {}
    for term in query_to_search:
        if inverted_index.df.get(term):
            ls_doc_freq = posting_lists_reader(inverted_index, term, anchor_prefix)
            for doc, freq in ls_doc_freq:
                result[doc] = result.get(doc, 0) + 1

    lst_doc = Counter(result).most_common()

    return lst_doc


def search_merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=100):
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

    for k, v in body_scores:
        if k not in result:
            result[k] = 0

        result[k] = result[k] + (text_weight * v)

    return sorted(result.items(), key=lambda x: x[1], reverse=True)[:N]




from flask import Flask, request, jsonify

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


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
    N = 10 #the number of docs i want to do the search on
    tokens = tokenize(query.lower())
    bestDocsBody = bm25(indexText,tokens,0.5,0.5,0.5,N)
    bestDocsTitle = query_get_tfidf_for_all_Title(indexTitle, tokens,N)
    bestDocs = search_merge_results(bestDocsTitle,bestDocsBody,0.5,0.5,N)
    res = get_docs_title_by_id(bestDocs)
    print(res)
    # END SOLUTION
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
    print("serchhhhhhhh")
    # tokens_after_filter = [term for term in tokens if indexText.df[term] < 300000 and term in indexText.df]
    bestDocs = query_get_top_N_tfidf(indexText,tokens,10)
    res = get_docs_title_by_id(bestDocs)
    # res = bestDocs
    # END SOLUTION
    print(res)
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
    print("1111111111111")
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    bestDocs = query_get_tfidf_for_all_Title(indexTitle, tokens)
    res = get_docs_title_by_id(bestDocs)

    # res = bestDocs
    print(res[:5])
    # END SOLUTION
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
    # # BEGIN SOLUTION
    # tokens = tokenize(query.lower())
    # bestDocs = query_get_tfidf_for_all_Anchor(indexAnchor,tokens) #here we don't want to filter the tokens becuase the titles are small not like text
    # res = get_docs_title_by_id(bestDocs)
    # res = bestDocs
    # END SOLUTION
    print(res)
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
    # dict_scores = pagerank['page rank'].to_dict()
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        res.append(page_ranks.get(id, 0))
    # END SOLUTION
    print(res)
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
    # BEGIN SOLUTION
    for id in wiki_ids:
        res.append(page_views.get(id, 0))
    # END SOLUTION
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
