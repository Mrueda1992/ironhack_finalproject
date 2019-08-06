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
file_unique_urls = file.copy()
file_unique_urls.drop_duplicates(subset='id', inplace=True)

#load nlp
nlp = spacy.load('en_core_web_sm')
parser = English()

#words tokenizer function
def words_tokenizer(text):
    tokens = parser(text)
    filtered_tokens = []
    for word in tokens:
        lemma = word.lemma_.lower().strip()
        if lemma not in STOP_WORDS and re.search ('^[a-zA-Z]+$', lemma):
            if lemma.endswith('ly'):
                continue
            filtered_tokens.append(lemma)
    return list(set(filtered_tokens))

#Vectorization of tokenized words
tfidf_vectorizer = TfidfVectorizer(min_df=0.15, max_df= 0.9, tokenizer=words_tokenizer)
terms_matrix = tfidf_vectorizer.fit_transform(file_unique_urls.text)
terms = tfidf_vectorizer.get_feature_names()

doc_term_matrix = terms_matrix.todense()
df_words = pd.DataFrame(doc_term_matrix, columns=terms, index=file_unique_urls.title)


#get UMAP out of tokenized words
def getUmap(dataset, n_components):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        umap = UMAP(n_components=n_components, n_epochs=15 , random_state=42).fit_transform(dataset)
        return pd.DataFrame(umap, columns=[f'emb_{i+1}' for i in range(n_components)])

umap_df = getUmap(df_words, 3)

#hdbscan to see cluster organization
def getClusters(umap):
    hdbscan = HDBSCAN(min_cluster_size=5)
    clusters = hdbscan.fit_predict(umap_df)
    return clusters

word_clusters = getClusters(umap_df)

#Plot a 2D scatter map to see UMAP reducion of dimensions
def scatterUMAP(dataset):
    plt.scatter(umap_df.emb_1, umap_df.emb_2, c=getClusters(umap_df))

#Plot a 3D map to see cluster representations of tokenized words
def clusters3D(dataset):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(umap_df.emb_1, umap_df.emb_2, umap_df.emb_3, marker='*', c=getClusters(umap_df), s=20)
    plt.show()

#Get name of articles within each cluster
def articlesinClusters(clusters):
    return file_unique_urls.title[word_clusters==clusters]

#Wordcloud
def wordcloud(text):
    wordcloud = WordCloud(max_font_size=50, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

#Wordcloud of all articles and for articles in each cluster
def clusterWordcloud(lista):
    return wordcloud(reduce((lambda x,y:x+y),lista))

print(clusterWordcloud(file_unique_urls.text.values))










