from nlp_processing import *

def main():
    file = pd.read_csv('text_classifier.csv', usecols=['id', 'title', 'text', 'Tags'])
    file_unique_urls = unique_urls(file)
    umap_df= getUmap(tfidf_matrix(file_unique_urls), 3)
    word_clusters= getClusters(umap_df)
    file_unique_urls['n_clusters'] = word_clusters
    clusters2D(umap_df, word_clusters)
    textWordcloud(file_unique_urls)
    clusterWordcloud(file_unique_urls)
    getDfCluster(-1, file_unique_urls, umap_df).T.sum(axis=1).sort_values(ascending=False)
    getDfCluster(0, file_unique_urls, umap_df).T.sum(axis=1).sort_values(ascending=False)
    getDfCluster(1, file_unique_urls, umap_df).T.sum(axis=1).sort_values(ascending=False)

if __name__== "__main__":
    main()
