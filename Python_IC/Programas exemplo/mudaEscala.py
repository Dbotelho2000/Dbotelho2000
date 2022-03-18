#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys

nomePastaImgEntrada = '/tmp/imagens/'
nomePastaResultado = '/tmp/res/'
nomesArqs = os.listdir( nomePastaImgEntrada )
print( 'nomesArqs:\n', nomesArqs )
for nomeUmArqEntr in nomesArqs:
    print( 'nomeUmArqEntr =', nomeUmArqEntr )
    ltNomeUmArq = nomeUmArqEntr.split( '.' )
    print( 'ltNomeUmArq =', ltNomeUmArq )
    matrImgEntr = cv2.imread( nomePastaImgEntrada + nomeUmArqEntr )
    ( altura, largura ) = matrImgEntr.shape[ :2 ]
    for fatorMudaEscala in np.arange( 0.8, 1.3, 0.2 ):
        fatorMudaEscala = round( fatorMudaEscala, 1 )
        print( 'fatorMudaEscala =', fatorMudaEscala )
        difAltura = abs( 1 - fatorMudaEscala ) * altura
        difLargura = abs( 1 - fatorMudaEscala ) * largura
        ( deslocaX, deslocaY ) = ( difLargura // 2, difAltura // 2 )
        if fatorMudaEscala < 1:
            matrTransformacao = np.float32( [
                [ fatorMudaEscala, 0, deslocaX ],
                [ 0, fatorMudaEscala, deslocaY ]
            ] )
        else:
            matrTransformacao = np.float32( [
                [ fatorMudaEscala, 0, -deslocaX ],
                [ 0, fatorMudaEscala, -deslocaY ]
            ] )
        nomeUmArqSaida = nomePastaResultado + \
            ltNomeUmArq[ 0 ] + '_escl_' + \
            str( fatorMudaEscala ).replace( '.', '_' ) + \
            '.' + ltNomeUmArq[ 1 ]
        print( 'nomeUmArqSaida =', nomeUmArqSaida )
        matrImgSaida = cv2.warpAffine( matrImgEntr, matrTransformacao,
                                       ( matrImgEntr.shape[ 1 ],
                                         matrImgEntr.shape[ 0 ] ) )
        cv2.imwrite( nomeUmArqSaida, matrImgSaida )
