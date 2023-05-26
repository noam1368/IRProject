# Project Name

Wikipedia Search Engine

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Methods](#methods)
- [Credits](#credits)
- [Creators](#creators)

## Introduction

The Wikipedia Search Engine is a program designed to provide users with an efficient and relevant search experience within the vast collection of articles available on Wikipedia. The engine utilizes a variety of different methods, including binary and cosine similarity algorithms, to filter out irrelevant pages and rank the most relevant ones based on user queries and page relevance.

## Features

- Fast and efficient search functionality.
- Filtering of irrelevant pages.
- Ranking of search results based on relevance.
- Support for various query types (single words, phrases, etc.).
- Integration with Wikipedia's API to access up-to-date information.

## Methods

The Wikipedia Search Engine utilizes various methods to enhance the search experience. Some of the methods implemented include:

1. **Binary Similarity Algorithm**: This algorithm calculates the similarity between the search query and each page by representing the query and page content as binary vectors. Pages with a higher number of matching terms have a higher similarity score.

2. **Cosine Similarity Algorithm**: This algorithm calculates the similarity between the search query and each page by treating the query and page content as vectors in a multi-dimensional space. The cosine of the angle between the query vector and page vector determines the similarity score.

3. **Stemming**: Stemming is a technique used to reduce words to their base or root form, allowing for more effective matching of similar words. It helps improve search accuracy by considering variations of words.

4. **Inverted Index**: An inverted index is a data structure that maps each unique word to the list of documents or pages in which it appears. It speeds up search queries by allowing efficient retrieval of documents containing specific words.

5. **BM25** (Best Match 25): BM25 is a ranking function commonly used in information retrieval systems. It calculates the relevance score between the search query and each document based on term frequencies and document lengths. BM25 considers factors like term frequency, document length, and document frequency to rank search results.

## Credits

The Wikipedia Search Engine utilizes the following libraries and frameworks:

- [Sparl](https://spark.apache.org/): A fast and general-purpose cluster computing system for Big Data processing.
- [NumPy](https://numpy.org/): A powerful library for numerical computing in Python.
- [Pandas](https://pandas.pydata.org/): A versatile data manipulation and analysis library.
- [NLTK (Natural Language Toolkit)](https://www.nltk.org/): A comprehensive toolkit for natural language processing in Python.

These tools have been instrumental in enabling efficient data processing, analysis, and natural language processing in the search engine.

## Creators

This Wikipedia Search Engine was created by [noam1368](https://github.com/noam1368) and [Ofek-Lutzky](https://github.com/Ofek-Lutzky) [. Thank you for your contribution to this project!

