from nlp_processing import *

def main():

    file = pd.read_csv('text_classifier.csv', usecols=['id', 'title', 'text', 'Tags'])
    file_unique_urls = unique_urls(file)
    umap_df= getUmap(tfidf_matrix(file_unique_urls), 3)
    word_clusters= getClusters(umap_df)
    file_unique_urls['n_clusters'] = word_clusters
    textWordcloud(file_unique_urls)
    clusterWordcloud(file_unique_urls)

if __name__== "__main__":
    main()

