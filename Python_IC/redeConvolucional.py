#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modo de uso:
# ./nomeDoPrograma -d nomeDirComImagens -t porcentExemplosTeste (1 a 99) \
#                  -e numEpocas -f numFiltrosPorCamada

# O programa espera encontrar, na pasta indicada, apenas arquivos com
# imagens de mesmo tamanho, cujos nomes sejam do tipo exNegat_XX.jpg
# ou exPosit_XX.jpg, contendo os exemplos negativos e positivos para
# treino e teste.

import getopt
import matplotlib.image as mpimg
import numpy as np
import os
import random
import sys

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical

optlist, args = getopt.gnu_getopt( sys.argv[ 1: ], 'd:e:f:t:' )
for( opcao, argumento ) in optlist:
    if opcao == '-d':
        dirImagens = argumento
    elif opcao == '-t':
        porcentExemplosTeste = float( argumento ) / 100
    elif opcao == '-e':
        numEpocas = int( argumento )
    elif opcao == '-f':
        numFiltrosPorCamada = int( argumento )
nomesArqs = os.listdir( dirImagens )
random.shuffle( nomesArqs )
#print( nomesArqs )
imagens = []
classes = []
for nomeArq in nomesArqs:
#    print( nomeArq )
    nomeClasse = nomeArq.split( '_' )[ 0 ]
#    print( nomeClasse )
    if nomeClasse == 'exNegat':
        classes.append( 0 )
    else:
        classes.append( 1 )
    umaImagem = mpimg.imread( dirImagens + nomeArq )
#    print( 'formato desta imagem:', umaImagem.shape, '\n' )
    imagens.append( umaImagem )
npImagens = np.array( imagens )
print( 'npImagens.shape =', npImagens.shape )
npImgEscala = npImagens / 255.0
npRotulos = np.array( classes )
print( 'npRotulos.shape =', npRotulos.shape )
numExemplos = npImgEscala.shape[ 0 ]
print( 'numExemplos =', numExemplos )
numLinhasImg = npImgEscala.shape[ 1 ]
print( 'numLinhasImg =', numLinhasImg )
numColunasImg = npImgEscala.shape[ 2 ]
print( 'numColunasImg =', numColunasImg )
tamUltDimensao = npImgEscala.shape[ -1 ]
print( 'tamanho da última dimensão de npImgEscala =', tamUltDimensao )
if tamUltDimensao == 3:
    numCanais = 3
else:
    numCanais = 1
    npImgEscala = npImgEscala.reshape( numExemplos,
                                       numLinhasImg, numColunasImg,
                                       numCanais )
numExemplosTeste = int( porcentExemplosTeste * numExemplos )
print( 'numExemplosTeste =', numExemplosTeste )
numExemplosTreino = numExemplos - numExemplosTeste
print( 'numExemplosTreino =', numExemplosTreino )
npImgTreino = npImgEscala[ 0:numExemplosTreino ]
print( 'npImgTreino.shape =', npImgTreino.shape )
npRotTreino = npRotulos[ 0:numExemplosTreino ]
print( 'npRotTreino.shape =', npRotTreino.shape )
npRotTreinoCateg = to_categorical( npRotTreino )
npImgTeste = npImgEscala[ numExemplosTreino: ]
print( 'npImgTeste.shape =', npImgTeste.shape )
npRotTeste = npRotulos[ numExemplosTreino: ]
print( 'npRotTeste.shape =', npRotTeste.shape )
npRotTesteCateg = to_categorical( npRotTeste )
# Criação da rede convolucional:
model = Sequential()
formaDoFiltro = ( 3, 3 )
# primeira camada
model.add( Conv2D( numFiltrosPorCamada, formaDoFiltro, activation = 'relu',
                   input_shape = ( numLinhasImg, numColunasImg, numCanais ) ) )
model.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )
# segunda camada
model.add( Conv2D( numFiltrosPorCamada, formaDoFiltro, activation = 'relu' ) )
model.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )
# flatten
model.add( Flatten() )
model.add( Dense( numFiltrosPorCamada, activation = 'relu' ) )
# softmax
model.add( Dense( units = 2, activation = 'softmax' ) )
model.compile( optimizer = "rmsprop",
               loss = 'categorical_crossentropy',
               metrics = [ 'accuracy' ] )
# treino:
model.fit( npImgTreino, npRotTreinoCateg,
           validation_data = ( npImgTeste, npRotTesteCateg ),
           epochs = numEpocas )
model.save( "modeloRedeConvolucional" )
# verificação:
previsoes = ( model.predict( npImgTeste ) )
print( 'type( previsoes ) =', type( previsoes ) )
print( 'previsoes.shape =', previsoes.shape )
print( 'npImgTeste.shape =', npImgTeste.shape )
print( 'type( nomesArqs ) =', type( nomesArqs ) )
print( 'len( nomesArqs ) =', len( nomesArqs ) )
nomesArqsTeste = nomesArqs[ numExemplosTreino: ]
print( 'type( nomesArqsTeste ) =', type( nomesArqsTeste ) )
print( 'len( nomesArqsTeste ) =', len( nomesArqsTeste ) )
print( '\nclassificações:\n' )
for ind in range( numExemplosTeste ):
    print( previsoes[ ind ], end = ' ' )
    print( nomesArqsTeste[ ind ] )
