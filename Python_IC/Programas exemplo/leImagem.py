#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Ao ser executado, deve-se fornecer um
#nome de arquivo de imagem na linha de comando.

import cv2
import sys

nomeArq = "C:/Users/Daniel/Desktop/Imagens/Com rebarba/CA1.PNG"
matrizImagem = cv2.imread( nomeArq )
print(matrizImagem.shape)
print(matrizImagem)
cv2.imshow( "imagem lida", matrizImagem )
cv2.waitKey( 0 )
