# _Inside tech's  hype_

Los avances tecnológicos han estado siempre ligados a la difusión y el intercambio del conocimiento. En la actualidad, el interés de la sociedad por temas como _blockchain_, criptomonedas, _machine learning_ o ciencia de datos crece a medida que ocupan cada vez más espacios del debate público. 

El propósito de este proyecto es arrojar luz sobre los conceptos que se utilizan en estos campos, con especial atención a las palabras clave o los 'tags' comunes que sirven para agrupar y categorizar estos temas de conversación. La cuestión es resolver preguntas tales como: cuando se habla de blockchain, ¿de qué o quiénes estamos hablando exactamente?

El proyecto consta de las siguientes fases: 

1. Scrapeo con las librerías Requests y BeautifulSoup de noticias publicadas en la página web www.hackernoon.com, un portal especializado en noticias sobre tecnología y startups.

2. Exploración y limpieza de datos llevada a cabo con la librería Pandas. También se han utilizado otras funcionalidades de Python como expresiones regulares (regex) para mejorar la calidad del texto.

3. Preprocesamiento de datos para NLP con Spacy (tokenización, lemmatización, stop words, etc.). El objetivo es extraer una lista de palabras 'tokenizadas' para la vectorización de textos.

4. Vectorización de texto con el algoritmo de Scikit-Learn TfidfVectorizer. 

5. Reducción de dimensionalidad y clusterización con UMAP y HDBSCAN

6. Creación de nube de palabras con Wordcloud
