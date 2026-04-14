from treinamento import carregar_dados, normalizar_dados_principais, calcular_distorcoes, calcular_numero_clusters, treinar_kmeans, salvar_modelo

from descrever_centroides import carregar_modelo, carregar_normalizador, descrever_clusters

from inferencia_cluster import normalizar_novo_dado, prever_cluster
import pandas as pd

# ====== carregar os dados ======
dados = carregar_dados('./HousingData.csv')
dados = dados.fillna(dados.mean())

media = dados.mean()

#print(dados)


# ====== normalizar os dados ======
dados_norm = normalizar_dados_principais(dados)
    
# transformar os dados normalizados em dataframe
dados_norm = pd.DataFrame(dados_norm, columns=dados.columns)

#print(dados_norm)


# ====== treinamento do modelo ======
distorcoes, valores_k = calcular_distorcoes(dados_norm)

clusters = calcular_numero_clusters(distorcoes, valores_k, dados_norm)

cluster_housingdata = treinar_kmeans(clusters, dados_norm)

salvar_modelo(cluster_housingdata)


# ====== descrever clusters ======
modelo_cluster = carregar_modelo('cluster_housingdata.pkl')

normalizador = carregar_normalizador('scaler_dados.pkl')

descrever_clusters(dados_norm, modelo_cluster, normalizador, media)


# ====== inferencia ======
novo_dado = pd.DataFrame([[
        0.30,   
        12.0,   
        5.50,   
        0,      
        0.48,   
        6.80,   
        45.0,   
        5.10,   
        4,      
        290,    
        17.0,   
        390.0,  
        7.50,   
        28.0    
    ]], columns=[
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
        'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ])

novo_dado_norm = normalizar_novo_dado(novo_dado, normalizador)

cluster_novo_dado = prever_cluster(modelo_cluster, novo_dado_norm)

print(f'\n>> Dados do novo imóvel: {novo_dado}')
print(f'\n>> Novo imóvel pertence ao Cluster: {cluster_novo_dado}')