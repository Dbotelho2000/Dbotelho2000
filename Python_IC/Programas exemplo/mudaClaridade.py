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
    matrImgEntr = cv2.imread( nomePastaImgEntrada + nomeUmArqEntr )
    ltNomeUmArq = nomeUmArqEntr.split( '.' )
    print( 'ltNomeUmArq =', ltNomeUmArq )
    # alpha – optional scale factor.
    # beta – optional delta added to the scaled values.
    beta = 0
    for alpha in np.arange( 0.6, 1.5, 0.2 ):
        alpha = round( alpha, 1 )
        print( 'alpha =', alpha )
        nomeUmArqSaida = ltNomeUmArq[ 0 ] + '_cl_' + \
            str( alpha ).replace( '.', '_' ) + \
            '.' + ltNomeUmArq[ 1 ]
        nomeUmArqSaida = nomePastaResultado + nomeUmArqSaida
        print( 'nomeUmArqSaida =', nomeUmArqSaida )
        matrImgSaida = cv2.convertScaleAbs( matrImgEntr,
                                            alpha = alpha,
                                            beta = beta )
        cv2.imwrite( nomeUmArqSaida, matrImgSaida )
