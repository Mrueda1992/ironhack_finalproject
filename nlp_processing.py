#import libraries
import re
import numpy as np
import pandas as pd
import spacy
import string
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from hdbscan import HDBSCAN
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from functools import reduce

#read csv
file = pd.read_csv('text_classifier.csv', usecols=['id', 'title', 'text', 'Tags'])

def unique_urls(file):
    file_unique_urls = file.copy()
    file_unique_urls.drop_duplicates(subset='id', inplace=True)
    return file_unique_urls

file_unique_urls = unique_urls(file)

#load nlp
nlp = spacy.load('en_core_web_sm')
parser = English()

#words tokenizer function
def words_tokenizer(text,repetidos=False):
    tokens = parser(text)
    filtered_tokens = []
    for word in tokens:
        lemma = word.lemma_.lower().strip()
        pos = word.pos_
        if lemma not in STOP_WORDS and re.search ('^[a-zA-Z]+$', lemma) and (pos == 'NOUN' and pos == 'PROPN'):
            filtered_tokens.append(lemma)
    if repetidos: return list(filtered_tokens)
    return list(set(filtered_tokens))

#Dataframe of vectorized and tokenized words
def tfidf_matrix(file):
    tfidf_vectorizer = TfidfVectorizer(min_df=0.15, max_df= 0.9, tokenizer=words_tokenizer, ngram_range=(1,2))
    terms_matrix = tfidf_vectorizer.fit_transform(file.text)
    doc_term_matrix = terms_matrix.todense()
    terms = tfidf_vectorizer.get_feature_names()
    df_words = pd.DataFrame(doc_term_matrix, columns=terms, index=file.title)
    return df_words

#get UMAP out of tokenized words
def getUmap(file, n_components):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        umap = UMAP(n_components=n_components, n_epochs=15 , random_state=42).fit_transform(file)
        return pd.DataFrame(umap, columns=[f'emb_{i+1}' for i in range(n_components)])

umap_df = getUmap(tfidf_matrix(file_unique_urls), 3)

#hdbscan to see cluster organization
def getClusters(umap):
    hdbscan = HDBSCAN(min_cluster_size=5)
    clusters = hdbscan.fit_predict(umap_df)
    return clusters


word_clusters = getClusters(umap_df)
file_unique_urls['n_clusters'] = word_clusters

#Plot a 2D scatter map to see UMAP reducion of dimensions
def clusters2D(dataset):
    plt.scatter(umap_df.emb_1, umap_df.emb_2, c=word_clusters)
    plt.show()

#Plot a 3D map to see cluster representations of tokenized words
def clusters3D(dataset):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(umap_df.emb_1, umap_df.emb_2, umap_df.emb_3, marker='*', c=word_clusters, s=20)
    plt.show()

#Get name of articles within each cluster
def articlesinClusters(clusters):
    return file_unique_urls.title[word_clusters==clusters]

def textinClusters(clusters):
    return file_unique_urls.text[word_clusters==clusters]

#Wordcloud
def wordcloud(text):
    wordcloud = WordCloud(max_font_size=50, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

#Wordcloud of all articles and for articles in each cluster
def textWordcloud(lista):
    return wordcloud(' '.join(words_tokenizer(reduce((lambda x,y:x+y),lista), repetidos=True)))

def clusterWordcloud(dataset):
    for x in set(dataset.n_clusters):
        textWordcloud(dataset[dataset.n_clusters==x].text)

#Find maximum occurrences of words in articles of each cluster by list of frequencies

def getDfCluster(cluster):
    return tfidf_matrix(file_unique_urls)[getClusters(umap_df)==cluster]

def getWordsFromCluster(dataset):
    return getWordsFromCluster(0).T.sum(axis=1).sort_values(ascending=False)

print(clusterWordcloud(word_clusters))















