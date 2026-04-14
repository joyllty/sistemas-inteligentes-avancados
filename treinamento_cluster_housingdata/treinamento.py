import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import  matplotlib.pyplot as plt
import math

'''
data set:
cada linha representa uma região/bairro com características médias
NAO ACHEI DADOS CATEGORICOS 

CRIM - taxa de criminalidade per capita por cidade
ZN - proporção de terrenos residenciais com lotes maiores que 25.000 pés² (áreas com casas grandes / bairros mais espaçosos)
INDUS - proporção de áreas industriais
CHAS - variável binária do Rio Charles
        1 → os imóveis ficam próximos ao rio
        0 → não ficam
NOX - concentração de óxidos de nítricos
RM - número médio de cômodos por residência
AGE - proporção de casas construídas antes de 1940 
DIS - distância ponderada até 5 centros de emprego de Boston 
RAD - índice de acesso a rodovias principais
TAX - taxa de imposto sobre propriedade (por $10.000)
PTRATIO - proporção aluno-professor (qualidade da educação — quanto menor, melhor)
B - fórmula relacionada à proporção de população negra
LSTAT - percentual da população de baixo status socioeconômico 
MEDV- valor mediano das casas (em milhares de dólares)
'''

def carregar_dados(path):
    '''
    devolve um data frame   
    '''
    dados = pd.read_csv(path, sep=',')

    return dados

def normalizar_dados_principais(dados):
    '''
    treina os dados principais, salva o normalizador de dados numéricos e aplica a normalização nos dados
    retorna um array

    precisa passar o nome do arquivo normalizador que será salvo, caso decida excluir a coluna medv
    '''
    scaler = MinMaxScaler()
    normalizador = scaler.fit(dados)
    # salvar o normalizador
    pickle.dump(normalizador, open('scaler_dados.pkl', 'wb'))

    dados_norm = normalizador.fit_transform(dados)

    return dados_norm

def calcular_distorcoes(dados_norm):
    '''
    calcular o numero ideal de centroides, e calcula o numero das distorções em relação a eles
    '''
    distorcoes = []
    # intervalo de pontos da reta da função distorcoes | numero de clusters
    K = range(1, 506) 

    for i in K:
        # treinando interativamente e aumentando o numero de clusters
        # testando quantidades de centroides para descobrir a ideal!!!!
        modelo = KMeans(n_clusters=i, random_state=42).fit(dados_norm)
        # calcular a distorção
        # media das distancias de cada ponto ao seu centroide
        # soma dos quadrados das distancias entre cada ponto e o centroide 
        distorcoes.append(
            sum(
                np.min(cdist(dados_norm, modelo.cluster_centers_, 'euclidean'), axis=1)/dados_norm.shape[0]
                )
            )   
    
    # fig, ax = plt.subplots()
    # ax.plot(K, distorcoes)
    # ax.set_xlabel('Número de Clusters')
    # ax.set_ylabel('Distorção')
    # ax.set_title('Método do Cotovelo')
    # ax.grid()
    #plt.show()

    return distorcoes, K

def calcular_numero_clusters(distorcoes, K, dados_norm):
    '''
    calcular o numero ideal de clusters, utilizando o metodo do cotovelo -> encontrar o ponto mais distante
    em linha reta da reta de clusters, isto é, o ponto onde adicionar mais um centroide reduzirá minimamente
    as distorções -> clusters bem definidos
    '''

    # definindo os pontos das retas
    x0 = K[0]
    y0 = distorcoes[0]
    xn = K[-1]
    yn = distorcoes[-1]

    distancias = []

    for i in range(len(distorcoes)):
        x= K[i]
        y= distorcoes[i]
        # FORMULA DISTANCIA PONTO-RETA (reta de dois pontos)
        numerador = abs(      # abs -> positivo independente do resultado
            (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
        )
        denominador = math.sqrt(
            (yn-y0)**2 + (xn-x0)**2
        )
        distancias.append(numerador/denominador)

    numero_clusters_otimo = K[distancias.index(np.max(distancias))]  
    print('\n>> Número ótimo de clusters = ', numero_clusters_otimo)


    return numero_clusters_otimo

def treinar_kmeans(numero_clusters_otimo, dados_norm):
    '''
    treinar o modelo de clusters
    '''
    cluster_housingdata = KMeans(n_clusters=numero_clusters_otimo, random_state=42).fit(dados_norm)

    return cluster_housingdata

def salvar_modelo(cluster_housingdata):
    '''
    salvar o modelo de clusters
    '''
    pickle.dump(cluster_housingdata, open('cluster_housingdata.pkl', 'wb'))

