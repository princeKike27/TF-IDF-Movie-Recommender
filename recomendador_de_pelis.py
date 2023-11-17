# -*- coding: utf-8 -*-
"""Recomendador de Películas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ikXnRcTw2pZWeNHa-FvtzxIq9qQMJfOH

# Recomendador de Películas

- Enrique Barón Gómez

"""

# importar modulos
import pandas as pd
import numpy as np

# json
import json

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity



# cargar data
df_pelis = pd.read_csv('https://raw.githubusercontent.com/princeKike27/Datasets-27/main/tmdb_5000_movies.csv')

df_pelis

# guardar primera peli del df
peli = df_pelis.iloc[0, :]


"""
![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Géneros de Cada Película
"""

# mirar generos de primera peli
peli['genres']

# convertir JSON a una lista de diccionarios
peli_generos = json.loads(peli['genres'])



"""- Podemos ver que la película Avatar tiene 4 Géneros

  - Cada uno de ellos se encuentra guradado dentro de la Key `name` de cada diccionario de la lista `peli_generos`
"""

# string para guardar generos de la peli
genero = ''

for el in peli_generos:
  # adicionar genero al string con un espacio entre ellos
    # si el genero contiene mas de una palabra estas se van a unir
  genero += ''.join(el['name'].split()) + ' '

genero

"""- Se observa que los 4 Géneros de Avatar son:

  - *Action*
  - *Adventure*
  - *Fantasy*
  - *ScienceFiction*

![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Palabras Clave de Cada Película
"""

# mirar palabras clave de peli
peli['keywords']

"""- Como se puede apreciar, las `keywords` de la películas estan en JSON por lo que es necesario convertirlos en una lista de diccionarios

  - Se va a utilizar el mismo metodo que se uso en la sección anterior para guardar las Palabras Clave de la Película en un `string`

  - Voy a crear una `Función` que toma como parámetro una película y me regresa un `string` con sus respectivos Géneros y Palabras Clave
"""

# crear funcion generos_y_palabras_clave
def generos_y_palabras_clave(pelicula):

  # convertir generos y palabras clave de peli a una lista de diccionarios
  generos = json.loads(pelicula['genres'])
  pa_claves = json.loads(pelicula['keywords'])

  # GENERO
  genero_string = ''

  for el in generos:
    # adicionar genero a string con un espacio entre ellos
    genero_string += ''.join(el['name'].split()) + ' '


  # PALABRAS CLAVE
  pa_claves_string = ''

  for el in pa_claves:
    # adicionar palabra clave a string con un espacio entre ellos
    pa_claves_string += ''.join(el['name'].split()) + ' '

  return f'{genero_string}{pa_claves_string}'

# probar funcion con peli
generos_y_palabras_clave(peli)

"""- El resultado de la función nos muestra los Géneros y las Palabras Clave de Avatar en un solo `string`

  - Este es el insumo que se va a utilizar para construir la Matriz `TF-IDF`  
"""

# crear columna con string de generos y palabras clave para cada peli
df_pelis['generos_paClaves'] = df_pelis.apply(generos_y_palabras_clave, axis=1)

# df con solo titulo y generos_paClaves
df_pelis = df_pelis[['title', 'generos_paClaves']]

# primeras 30 pelis
df_pelis.head(30)

# mirar si hay peliculas que no tienen generos ni palabras clave
idx_vacios = list(np.where(df_pelis.generos_paClaves == '')[0])

idx_vacios

# no tener en cuenta pelis con idx_vacios
df_pelis = df_pelis.drop(idx_vacios)

df_pelis

"""![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Mapping de Palabra a Index

- Voy a Mapear cada Género y Palabra Clave Única en el *Corpus* a un Índice

  - *Corpus* ⟶ Las Películas que hay en el dataset
  - *Token* ⟶ Género o Palabra Clave
  - Se va a crear un Diccionario ⟶ `key: token, Value: índice`
"""

# empezar indice en 0
idx = 0

# crear diccionario
token_A_idx = {}

# lista para guardar indices de los tokens de cada peli
tokenized_pelis = []

# iterar sobre cada peli
for peli in df_pelis['generos_paClaves']:

  # split() >> lista de generos y palabras clave de peli
  tokens = peli.split()
  # lista para guardar los indices de la peli
  peli_idx = []

  for token in tokens:

    # pasar token a minuscula
    token = token.lower()

    # check si el token esta en el diccionario
    if token not in token_A_idx:
      # mapear token a idx
      token_A_idx[token] = idx
      # incrementar valor de idx
      idx += 1

    # append el indice del token
    peli_idx.append(token_A_idx[token])

  # append los indices de los generos y palabras clave de la peli
  tokenized_pelis.append(peli_idx)

# mirar token_A_idx
token_A_idx.items()

"""- El diccionario `token_A_idx` contiene los Tokens (Géneros y Palabras Clave) Únicos de las Películas del dataset

![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Matriz de Vocabulario y de Frecuencia de Terminos
"""

# número de documentos (Peliculas)
N = len(df_pelis)

#print(f'Número de Documentos (Películas): {N}')

# Vocabulario >> numero de tokens unicos
V = len(token_A_idx)

#print(f'Vocabulario: {V}')

# Term-Frequency Matrix >> (N, V)
tf = np.zeros((N, V))

# mirar dimensiones
##print(f'tf dimension: {tf.shape}', '\n')

tf

# popular la Matriz tf

# iterar sobre tokenized_pelis >> contiene los indices de los tokens de cada peli
for i, peli_idx in enumerate(tokenized_pelis):

  # por cada indice en la peli i
  for j in peli_idx:
    # ir a la columna j y adicionar 1
    tf[i, j] += 1

# mirar primeras 100 columnas de la 4ta pelicula del dataset
tf[3, 0:100]

"""- **Un Token va a tener un Valor $> 0$ si corresponde al Género o Palabras Claves de la Película**

![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Calcular TF-IDF

- Inverse Document Frequency ⟶ Valor para cada Token en el Vocabulario

  <br>

  $$ IDF = \log (\frac{N}{\text{Document Frequency}}) $$

  <br>
"""

# Document Frequency >> suma sobre cada token
document_freq = np.sum(tf > 0, axis=0)

# mirar dimensiones
#print(f'Document Frequency dimension: {document_freq.shape}', '\n')

document_freq

"""- El Document Frequency es un Vector que tiene el tamaño del Vocabulario

  - El Primer Token aparece 1156 veces en las Películas
  - El Tercer Token aparece 219 veces en las Películas
"""

# Inverse Document Frequency (IDF)
idf = np.log(N / document_freq)

# mirar dimension
#print(f'idf shape: {idf.shape}', '\n')

idf

# TF-IDF
tf_idf = tf * idf

# mirar dimensiones
#print(f'tf_idf dimension: {tf_idf.shape}', '\n')

# primeras 100 columnas de la 4ta pelicual
tf_idf[3, 0:100]

"""- **TF-IDF hace un énfasis en los Tokens poco comunes del Corpus que se esta analizando**
  
  - Reduce la Importancia de los Géneros y Palabras Claves que son comunes entre todas las Películas

![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Mapping del Título de la Película al Índice

- Es necesario Mapear el Título de la Película con su respectivo Índice ya que, la Matriz TF-IDF consta de 4777 filas, donde cada fila corresponde al índice de una película en el dataset
"""

# crear serie >> peli_A_idx
peli_A_idx = pd.Series(df_pelis.index, index=df_pelis['title'])

peli_A_idx

# indice de la peli >> Mortal Kombat
peli_A_idx['Mortal Kombat']

# tf_idf de Mortal Kombat
mortal_kombat = tf_idf[2100, :]

# reshape >> (1, -1)
mortal_kombat = mortal_kombat.reshape(1, -1)

#print(f'Dimension del tf_idf de Mortal Kombat: {mortal_kombat.shape}', '\n')

mortal_kombat

"""- El `TF-IDF` de la película `Mortal Kombat` es un vector de Longitud $9,767$ que contiene los puntajes de cada token

![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Similitud de Coseno

- Voy a utilizar la `Similitud de Coseno` para encontrar las Películas que son Más Similares a `Mortal Kombat`

  - **Entre mayor sea la Similitud de Coseno entre dos películas estas son más similares y por ende, se pueden recomendar**
"""

# similitud de coseno entre Mortal Kombat y las demas peliculas
scores_mortal_kombat = cosine_similarity(mortal_kombat, tf_idf)

# mirar dimension
#print(f'puntajes_mortal_kombat dimension: {scores_mortal_kombat.shape}', '\n')

scores_mortal_kombat

# flatten array >> (4777, )
scores_mortal_kombat = scores_mortal_kombat.flatten()

# mirar dimension
#print(f'scores_mortal_kombat dimension: {scores_mortal_kombat.shape}')

scores_mortal_kombat

# 10 pelis con la similitud de coseno mas alta con respecto a Mortal Kombat
scores_mortal_kombat[(-scores_mortal_kombat).argsort()][1:11]

"""- **Entre más Cercano a $1$ es la Similitud Coseno ⟶ La Película es más similar a Mortal Kombat y se puede Recomendar**"""

# ordenar scores DESC y regresar los indices de las peliculas
(-scores_mortal_kombat).argsort()


"""
![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Índices y Títulos de las Películas Recomendadas

- Se van a Recomendar las 5 Películas con los Puntajes más Altos de Similitud de Coseno de una película que le haya gustado al Usuario
"""

# indices de los top 5 scores
idx_recomendados = (-scores_mortal_kombat).argsort()[1:6]

idx_recomendados

# pelis a recomendar
#print('Peliculas Recomendas si te gusto Mortal Kombat:', '\n')

"""
for i in idx_recomendados:
  print(df_pelis['title'].iloc[i])
"""
  
"""![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

## Función para Recomendar Películas

- Voy a tomar lo construido en los apartados anteriores para crear una función que toma como parametros una pelicula que le gusto al usuario y regresa las recomendaciones para esa pelicula
"""

"""
# funcion para recomendar peliculas
def recomendar_pelis(peli):

  # si la peli esta en el dataset
  try:

    # indice de peli
    idx_peli = peli_A_idx[peli]

    # tf_idf de peli
    tfidf_peli = tf_idf[idx_peli, :]
    # reshape >> (1, -1)
    tfidf_peli = tfidf_peli.reshape(1, -1)

    # similitud de coseno de peli
    scores_peli = cosine_similarity(tfidf_peli, tf_idf)
    # flatten scores_peli >> (-1, )
    scores_peli = scores_peli.flatten()

    # ordenar scores DESC y guardar indices de top 5 scores
    top_5_scores_idx = (-scores_peli).argsort()[1:6]

    # print resultados
    #print(f'Películas Recomendadas si te gusto {peli}:', '\n')

    #for i in top_5_scores_idx:
      # titulo de pelicula en el indice i
      #print(df_pelis['title'].iloc[i])


  except:
    print(f'La película {peli} no se encuentra en la Base de Datos :(')

# recomendaciones para Superman Returns
recomendar_pelis('Superman Returns')

# recomendaciones para Robin Hood
recomendar_pelis('Robin Hood')

# recomendaciones para The Chronicles of Narnia: Prince Caspian
recomendar_pelis('The Chronicles of Narnia: Prince Caspian')

# recomendaciones para A Haunting in Venice
recomendar_pelis('A Haunting in Venice')

# recomendaciones para The Hobbit: The Desolation of Smaug
recomendar_pelis('The Hobbit: The Desolation of Smaug')

"""

"""![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

![purple-divider](https://user-images.githubusercontent.com/7065401/52071927-c1cd7100-2562-11e9-908a-dde91ba14e59.png)
"""