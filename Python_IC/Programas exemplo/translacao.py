#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys

nomePastaImgEntrada = 'C:/Users/Daniel/Desktop/Imagens/Com rebarba/'
nomePastaResultado = 'C:/Users/Daniel/Desktop/Imagens/RES_COM/'
nomesArqs = os.listdir( nomePastaImgEntrada )
print( 'nomesArqs:\n', nomesArqs )
for nomeUmArqEntr in nomesArqs:
    print( 'nomeUmArqEntr =', nomeUmArqEntr )
    ltNomeUmArq = nomeUmArqEntr.split( '.' )
    print( 'ltNomeUmArq =', ltNomeUmArq )
    matrImgEntr = cv2.imread( nomePastaImgEntrada + nomeUmArqEntr )
    for tamTranslacaoHorizontal in range( -50, 51, 50 ):
        print( 'tamTranslacaoHorizontal =', tamTranslacaoHorizontal )
        for tamTranslacaoVertical in range( -50, 51, 50 ):
            print( 'tamTranslacaoVertical =', tamTranslacaoVertical )
            matrTransformacao = np.float32( [
                [ 1, 0, tamTranslacaoHorizontal ],
                [ 0, 1, tamTranslacaoVertical ]
            ] )
            matrImgSaida = cv2.warpAffine( matrImgEntr, matrTransformacao,
                                           ( matrImgEntr.shape[ 1 ],
                                             matrImgEntr.shape[ 0 ] ) )
            nomeUmArqSaida = nomePastaResultado + \
                ltNomeUmArq[ 0 ] + '_hrz_' + \
                str( tamTranslacaoHorizontal ) + \
                '_vrt_' + str( tamTranslacaoVertical ) + \
                '.' + ltNomeUmArq[ 1 ]
            print( 'nomeUmArqSaida =', nomeUmArqSaida )
            matrImgSaida = cv2.warpAffine( matrImgEntr, matrTransformacao,
                                           ( matrImgEntr.shape[ 1 ],
                                             matrImgEntr.shape[ 0 ] ) )
            cv2.imwrite( nomeUmArqSaida, matrImgSaida )
