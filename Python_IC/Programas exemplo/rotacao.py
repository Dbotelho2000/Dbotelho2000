#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys

nomePastaImgEntrada = 'C:/Users/Daniel/Desktop/Imagens/RES_COM/'
nomePastaResultado = 'C:/Users/Daniel/Desktop/Imagens/RES2/'
nomesArqs = os.listdir( nomePastaImgEntrada )
print( 'nomesArqs:\n', nomesArqs )
for nomeUmArqEntr in nomesArqs:
    print( 'nomeUmArqEntr =', nomeUmArqEntr )
    ltNomeUmArq = nomeUmArqEntr.split( '.' )
    print( 'ltNomeUmArq =', ltNomeUmArq )
    matrImgEntr = cv2.imread( nomePastaImgEntrada + nomeUmArqEntr )
    ( altura, largura ) = matrImgEntr.shape[ :2 ]
    ( centroX, centroY ) = ( largura // 2, altura // 2 )
    for anguloRotacaoGraus in range( -15, 16, 15 ):
        print( 'anguloRotacaoGraus =', anguloRotacaoGraus )
        matrTransformacao = cv2.getRotationMatrix2D( ( centroX, centroY ),
                                                     anguloRotacaoGraus, 1.0 )
        matrImgSaida = cv2.warpAffine( matrImgEntr, matrTransformacao,
                                       ( matrImgEntr.shape[ 1 ],
                                         matrImgEntr.shape[ 0 ] ) )
        nomeUmArqSaida = nomePastaResultado + \
            ltNomeUmArq[ 0 ] + '_rot_' + \
            str( anguloRotacaoGraus ) + \
            '.' + ltNomeUmArq[ 1 ]
        print( 'nomeUmArqSaida =', nomeUmArqSaida )
        cv2.imwrite( nomeUmArqSaida, matrImgSaida )
