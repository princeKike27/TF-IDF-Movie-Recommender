# Importar Modulos
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as sk
from flask import Flask, request, jsonify, render_template, url_for
from recomendador_de_pelis import tf_idf, peli_A_idx, df_pelis


'''Flask
Web Framework liviano para construir aplicaciones web
'''

# inicializar flask
app = Flask(__name__)

# guardar tf_idf, peli_A_idx y df_pelis
tf_idf = tf_idf
peli_A_idx = peli_A_idx
df_pelis = df_pelis

# lista de peliculas
lista_pelis = df_pelis.title

'''***********************************************************************'''
''' ENDPOINTS & ROUTINES
HTTP >> request-response protocol to connect between a client & server
GET >> request data from a specified source (ex: web page form)
POST >> data sent to the server to create/update a resource (ex: web page element)
'''

'''HOME'''
@app.route('/', methods=['GET', 'POST'])
def home():
    # render home con lista_pelis
    return render_template('home.html', lista_pelis=lista_pelis)


'''RECOMENDAR PELIS'''
@app.route('/recomendar', methods=['POST'])
def recomendar_pelis():
    # guardar peli del usuario
    peli_usuario = request.form.get('peli')
    print('\n')
    print(f'Peli favorita del usuario: {peli_usuario}')

    # try >> si la peli esta en lista_pelis
    try:
        # indice de la peli
        peli_idx = peli_A_idx[peli_usuario]
        print(f'Indice de peli: {peli_idx}')

        # tf-idf de peli
        peli_tfidf = tf_idf[peli_idx, :]
        # reshape a 1, -1 >> (1, V)
        peli_tfidf = peli_tfidf.reshape(1, -1)
        # mirar dimension
        print(f'peli_tfidf dimension: {peli_tfidf.shape}')    

        # calcular cosine_similarity entre peli y demas peliculas
        scores_peli = sk.cosine_similarity(peli_tfidf, tf_idf)
        # flatten >> Vector de longitud len(lista_pelis)
        scores_peli = scores_peli.flatten()
        # mirar dimensiones
        print(f'scores_peli dimensiones: {scores_peli.shape}')

        # ordenar scores DESC y guardar indices de los 5 scores mas altos
        top_5_scores_idx = (-scores_peli).argsort()[1:6]
        print(f'top_5_scores_idx: {top_5_scores_idx}')

        # lista de pelis a recomendar
        pelis_recomendadas = [df_pelis['title'].iloc[i] for i in top_5_scores_idx]
        print(f'pelis_recomendads: {pelis_recomendadas}', '\n')


        return render_template('reco.html', peli_usuario=peli_usuario, pelis_recomendadas=pelis_recomendadas)

    # si peli no esta en lista_pelis
    except:
        print(f'{peli_usuario} No se encuentra en la DB', '\n')
        return render_template('error.html', peli_usuario=peli_usuario)


'''Run Server'''
if(__name__) == '__main__':
    print('*' * 100)
    print('Empezando Python Flask Server para Pelis Recommender ...', '\n')

    # check que se han importado tf_idf, peli_A_idx, df_pelis
    print(f'tf_idf dimension: {tf_idf.shape}, type: {type(tf_idf)}')
    print(f'peli_A_idx dimension: {peli_A_idx.shape}, type: {type(peli_A_idx)}')
    print(f'df_pelis dimension: {df_pelis.shape}, type: {type(df_pelis)}', '\n')

    # check dimension de lista_pelis
    print(f'lista_pelis dimension: {len(lista_pelis)}, Primeras 10 pelis:', '\n')
    print(lista_pelis[:10], '\n')

    # run server
    app.run(debug=True)