## InformationRetrieval

The aim of the project is to build a search engine which handles different queries effectively by retrieving the highly ranked documents from a corpus that has been built having 418 documents from the Environmental News NLP archive.

We also retrieve the data, compile and compare metrics with a distributed, open source search engine named Elasticsearch.

This search engine built handles three kinds of query:
1. Free text query
2. Phrase query
3. Wildcard Query

The main functionality of our project is to build a search engine which exhibits high accuracy in fetching the required documents upon querying.

This search engine is constructed through the following steps:
1. Search for the terms in the query
2. Create Postings list
3. Fill the Inverted Index
4. Rank the pages
5. Retrieve the data from the dictionary
6. Calculate query response time
7. Measure the efficiency using precision, recall, F measure.

Assignment includes the following parts.
1. Token generation :
The dataset contains 417 csv files. We extracted the data from the dataset and
did preprocessing such as expanding contractions, removal of punctuations and
stopwords, converting the case to lower and stemming. We used
word_tokenize(), WordNetLemmatizer(), PorterStemmer() functions for the
same.
2. Index Creation :
Created three indexes - Inverted Index, Positional Index for phrase query ,
K-gram(3-gram) index for wildcard query.
3. Intersection of posting lists of positional indexes and k-gram indexes using
intersection algorithm.
4. Computation of Scores
5. Retrieval of Top K documents:
To retrieve the top 20 documents from the corpus which have the highest
score, we have used a heap queue.
6. Comparison of the search engine with elastic search search engine.
7. Computation of evaluation metrics such as precision, recall, F-scores to
measure the efficiency of our search engine.
