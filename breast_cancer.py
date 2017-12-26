#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:51:37 2017

@author: igorpedromartins
"""

'''
    Trabalhando com dados reais sobre câncer, para que assim seja descoberto 
    quem pode adquirir câncer ou não
'''

import numpy as np

# Importando dataset
from sklearn import datasets 

'''
 Variaveis trabalhadas no projeto
'''


base_dados = datasets.load_breast_cancer()



# Padrões a ser aprendidos

entradas = base_dados.data


# Resultado que o algoritmo deverá chegar

valor_saida = base_dados.target

# Criando array vazia para converter dados simples em uma matriz
saidas = np.empty ([569, 1], dtype = int)

for i in range (569):
    
    saidas [i] = valor_saida[i]

# O Número 3 representa a quantidade de camada escondida
pesos1 = 2 * np.random.random ((30, 5)) - 1

# peso 2 parte da camada escondida e vai para a saida da rede neural
pesos2 = 2 * np.random.random ((5, 1)) - 1


epocas = 10000

momento = 1
taxa_aprendizagem = 0.3



# Funcões do Projeto

# Função Soma
def somatorio (entrada, peso):
    
    return np.dot (entrada, peso)

# Função Ativação
def sigmoid (soma):
    
    return 1 / (1 + np.exp(-soma))

# Derivada Parcial
def derivada_parcial (sigmoide): 
    
    return sigmoide * (1 - sigmoide)

# Delta
def delta (erro, derivada):
    
    return erro * derivada




for rodada in range (epocas):
    
    camada_entrada = entradas
    
    # Primeira Sinapse
    soma_sinapse0 = somatorio (camada_entrada, pesos1) # Somatorio
    camada_oculta = sigmoid (soma_sinapse0) # Função de ativação
    
    # Segunda Sinapse
    soma_sinapse1 = somatorio (camada_oculta, pesos2) # Somatorio dois
    camada_saida = sigmoid (soma_sinapse1)  # Funcão de ativação dois

    # Calculando o erro  
    erro = saidas - camada_saida
    media_absoluta = np.mean (np.abs (erro))
    #Porcentagem de acerto do algoritmo   
    print ('ERRO: ', str (media_absoluta))
    
    '''    
        Cálculos para melhorar o modelo de aprendizagem 
    '''

    derivada_saida = derivada_parcial (camada_saida)
    delta_saida = delta (erro, derivada_saida)
    

    # Fazendo método de função, utilizada para calcular os neuronios da camada escondida
    #    delta_oculta = derivada * pesos * delta
    # Forward

    matriz_transposta = pesos2.T
    resultado_produto = delta_saida.dot (matriz_transposta) # nome da variavel pode ser chamada de result_multiplicacao
    delta_camada_oculta = resultado_produto * derivada_parcial (camada_oculta)
    
    # Backpropagation
    camada_oculta_transposta = camada_oculta.T
    peso_novo = camada_oculta_transposta.dot(delta_saida)
    pesos2 = (pesos2 * momento) + (peso_novo * taxa_aprendizagem)
    
    camada_entrada_transposta = camada_entrada.T
    peso_novo1 = camada_entrada_transposta.dot (delta_camada_oculta)
    pesos1 = (pesos1 * momento) + (peso_novo1 * taxa_aprendizagem)
    
    
    
    