from nlp_processing import *
from parsing_project import parse

def main():
    file = pd.read_csv('text_classifier.csv', usecols=['id', 'title', 'text', 'Tags'])
    file_unique_urls = unique_urls(file)
    vector_matrix = tfidf_matrix(file_unique_urls)
    umap_df = getUmap(vector_matrix, 3)
    word_clusters = getClusters(umap_df)
    file_unique_urls['n_clusters'] = word_clusters
    clusterWordcloud(file_unique_urls)


if __name__== "__main__":
    main()

