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

#todo change the paths
#this is the read of the indexes we made before in the storage of the cloud
indexTiltePath="./indexes_bucket/indexTitle.parquet/"
indexTextPath="./indexes_bucket/indexText.parquet/"
indexAnchorPath="./indexes_bucket/indexTextPath/" #todo
postingsGcpPath="./indexes_bucket/postings_gcp/"
# indexPageViews = "./indexes_bucket/postings_gcp/" #todo


indexTitle= InvertedIndex.read_index(indexTiltePath, "index_title")
indexText=InvertedIndex.read_index(indexTextPath, "index")#todo
indexAnchor=InvertedIndex.read_index(indexAnchorPath, "index_anchor")

#we store the values of all the wikipedia docs
with open(indexTiltePath, 'rb') as pkl:
    docs_titles=pickle.load(pkl) # titlesDocs todo delete this comment

# with open(indexPageViews, 'rb') as pkl: todo delete this comment
#     pageviews_wikidamp = pickle.loads(pkl.read())

with open(postingsGcpPath, 'rb') as pkl:#nf todo delete this comment
    # postings_norm=pickle.load(pkl)
    DL = pickle.load(pkl)

############################################################################

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





def query_get_top_N_tfidf_just_for_Title(inverted_index, query_to_search):
    """
    Args:
        query_to_search:
        index:

    Returns:

    """
    result = {}  # result[doc_id] = score now it will be according to the number of frequency of the word inside the title sum of doc_tf.
    for token in query_to_search:
        term = inverted_index.df.get(token)
        if term:
            docs = posting_lists_reader(inverted_index,token)
            for doc_id, doc_tf in docs:
                result[doc_id] = result.get((doc_id, token), 0) + doc_tf

    return result.items()



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
    tokens = tokenize(query.lower())
    #we are doing the assumption that the search on the text is inuf to tell what is the best 100
    #todo maybe think about weight to the title and text like ass 4
    tokens_after_filter = [token for token in tokens if token in indexText.df[token] < 250000 and indexText.df] #todo change the number according to number we see warking good
    bestDocs = query_get_top_N_tfidf(indexText,tokens_after_filter,100)
    res = get_docs_title_by_id(bestDocs)
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
    tokens_after_filter = [token for token in tokens if token in indexText.df[token] < 250000 and indexText.df] #todo change the number according to number we see warking good
    bestDocs = query_get_top_N_tfidf(indexText,tokens_after_filter,100)
    res = get_docs_title_by_id(bestDocs)
    # END SOLUTION
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
    # BEGIN SOLUTION
    tokens = tokenize(query.lower())
    bestDocs = query_get_top_N_tfidf_just_for_Title(indexTitle,tokens) #here we don't want to filter the tokens becuase the titles are small not like text
    bestDocs.sort(key = lambda x:x[1], reverse = True)
    res = bestDocs
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
    # BEGIN SOLUTION
    
    # END SOLUTION
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
    # BEGIN SOLUTION

    # END SOLUTION
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

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
