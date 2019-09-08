from nlp_processing import *
import argparse
import sys

def keywords(title, df):
    return df.iloc[title:].sort_values(ascending=False)[:10]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--title', type=str, help='Imprime los  tags de un artículo', action='store_true')
    args = parser.parse_args()
    if args.title:
        print('Keywords del artículo: {}'.format(keywords(args.title, tfidf_matrix(file_unique_urls))))
    else:
        print('Error: inserta un título válido.')






