import re
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import spacy
from hdbscan import HDBSCAN
from umap import UMAP
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from functools import reduce
from time import sleep

#read csv
def unique_urls(file):
    file.drop_duplicates(subset='id', inplace=True)
    return file

#load nlp and words tokenizer function
def words_tokenizer(text,repetidos=False):
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(text)
    filtered_tokens = []
    for word in tokens:
        lemma = word.lemma_.lower().strip()
        pos = word.pos_
        if lemma not in STOP_WORDS and re.search ('^[a-zA-Z]+$', lemma) and pos == 'NOUN':
            if lemma == 'datum':
                continue
            elif lemma.endswith('ly') or lemma == 'will':
                continue
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

#hdbscan to see cluster organization
def getClusters(umap):
    hdbscan = HDBSCAN(min_cluster_size=5)
    clusters = hdbscan.fit_predict(umap)
    return clusters

#Plot a 2D scatter map to see UMAP reducion of dimensions
def clusters2D(dataset, params):
    plt.scatter(dataset.emb_1, dataset.emb_2, c=params)
    plt.show(block=False)
    plt.pause(3)
    plt.close(1)

#Get name of articles within each cluster
def articlesinClusters(file, clusters):
    return file.title[word_clusters==clusters]

def textinClusters(file, clusters):
    return file.text[word_clusters==clusters]

#Wordcloud
def wordcloud(text):
    wordcloud = WordCloud(max_font_size=50, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show(block=False)
    plt.pause(3)
    plt.close(1)

#Wordcloud of all articles and for articles in each cluster
def textWordcloud(lista):
    return wordcloud(' '.join(words_tokenizer(reduce((lambda x,y:x+y), lista.text), repetidos=True)))

def clusterWordcloud(dataset):
    for x in set(dataset.n_clusters):
        textWordcloud(dataset[dataset.n_clusters==x])

#Get dataframe of token words for each cluster
def getDfCluster(cluster, file, umap):
    return tfidf_matrix(file)[getClusters(umap)==cluster]






