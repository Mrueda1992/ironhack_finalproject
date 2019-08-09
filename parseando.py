from nlp_processing import *
import argparse

def keywords(title, df):
    return df.iloc[title:].sort_values(ascending=False)[:10]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--title', help='Imprime las palabras clave de un artículo.', action='store_true')
    return parser.parse_args()




